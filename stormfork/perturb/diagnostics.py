from collections.abc import Mapping
from math import sqrt

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from stormfork.grid import Grid, great_circle_km
from stormfork.perturb._contracts import checked_names
from stormfork.perturb.base import Taper

INITIAL_STEPS = 2
FORECAST_INTERVAL = np.timedelta64(6, "h")


class DiagnosticError(ValueError):
    """A balance diagnostic is undefined for the supplied forecast states."""


def _annulus_rms(ds: xr.Dataset, name: str, annulus: NDArray[np.bool_], source: str) -> float:
    if name not in ds.data_vars:
        msg = f"{source}: missing pressure variable {name}"
        raise DiagnosticError(msg)
    pressure = ds[name]
    if "batch" in pressure.dims:
        if pressure.sizes["batch"] != 1:
            msg = f"{source}: {name} requires a singleton batch dimension"
            raise DiagnosticError(msg)
        pressure = pressure.squeeze("batch", drop=True)
    if "member" in pressure.dims:
        if pressure.sizes["member"] == 0:
            msg = f"{source}: {name} requires at least one member"
            raise DiagnosticError(msg)
        pressure = pressure.isel(member=0)
    if set(pressure.dims) != {"time", "lat", "lon"}:
        msg = (
            f"{source}: {name} requires time, lat, lon dimensions, "
            "optional member and singleton batch"
        )
        raise DiagnosticError(msg)
    if pressure.sizes["time"] < INITIAL_STEPS:
        msg = f"{source}: {name} requires at least two forecast steps along time"
        raise DiagnosticError(msg)
    values = np.asarray(
        pressure.isel(time=slice(0, INITIAL_STEPS)).transpose("time", "lat", "lon").values,
        dtype=np.float64,
    )
    tendency = values[1][annulus] - values[0][annulus]
    if not np.all(np.isfinite(tendency)):
        msg = f"{source}: {name} tendency must be finite throughout the taper annulus"
        raise DiagnosticError(msg)
    return sqrt(float((tendency * tendency).mean()))


def _first_interval(ds: xr.Dataset, source: str) -> np.timedelta64 | None:
    if "time" not in ds.coords:
        return None
    time = ds.coords["time"]
    if time.dims != ("time",) or time.size < INITIAL_STEPS:
        msg = f"{source}: time requires at least two forecast steps on a one-dimensional coordinate"
        raise DiagnosticError(msg)
    if time.dtype.kind not in ("M", "m"):
        msg = f"{source}: time interval requires datetime or timedelta coordinates for six hours"
        raise DiagnosticError(msg)
    return time.values[1] - time.values[0]


def _check_intervals(fork: xr.Dataset, baseline: xr.Dataset) -> None:
    fork_interval = _first_interval(fork, "fork")
    baseline_interval = _first_interval(baseline, "baseline")
    if (
        fork_interval is not None
        and baseline_interval is not None
        and fork_interval != baseline_interval
    ):
        msg = "fork and baseline first time intervals must match and represent six hours"
        raise DiagnosticError(msg)
    for source, interval in (("fork", fork_interval), ("baseline", baseline_interval)):
        if interval is not None and interval != FORECAST_INTERVAL:
            msg = f"{source}: first time interval must represent six hours"
            raise DiagnosticError(msg)


def adjustment_ratio(
    fork: xr.Dataset,
    baseline: xr.Dataset,
    taper: Taper,
    *,
    names: Mapping[str, str] | None = None,
) -> float:
    """Return the dimensionless fork/baseline RMS pressure-tendency ratio.

    Pressure uses Pa on named time, lat, lon axes in any order, with an optional
    member axis selected at index 0 and a singleton batch axis squeezed away.
    The first two time entries must represent
    the initial state and its six-hour forecast; later entries are ignored.
    When time coordinates exist, they must be datetime or timedelta coordinates
    with a six-hour first interval, matching between fork and baseline. Without
    time coordinates the caller supplies the six-hour layout.
    Both grids must match: ascending, equally spaced lat/lon coordinates with
    matching spacing and global longitudes covering [0, 360) from zero.
    Optional names map mean_sea_level_pressure to the dataset variable name.

    RMS is unweighted over cell centres at great-circle distances from radius/2
    through radius, inclusive, in km from the taper centre in degrees. A ratio
    near 1 means similar pressure-change magnitude; a larger ratio flags excess
    change for investigation. No threshold or physical balance interpretation
    has been calibrated against forecasts.
    Inputs are never modified.

    Raises:
        DiagnosticError: Pressure layout is invalid, the annulus is empty, its
            tendency is nonfinite, or the baseline RMS tendency is zero.
        KeyError: An explicitly mapped variable is missing from either dataset.
        ValueError: A names key is unknown. Either grid violates its contract or the grids differ.
    """
    aliases = checked_names(fork, names, ("mean_sea_level_pressure",))
    checked_names(baseline, names, ("mean_sea_level_pressure",))
    _check_intervals(fork, baseline)
    grid = Grid.from_dataset(fork)
    baseline_grid = Grid.from_dataset(baseline)
    if not np.array_equal(grid.lat, baseline_grid.lat) or not np.array_equal(
        grid.lon, baseline_grid.lon
    ):
        msg = "fork and baseline lat/lon coordinates must match"
        raise ValueError(msg)
    distance = great_circle_km(taper.centre_lat, taper.centre_lon, grid.lat[:, None], grid.lon)
    annulus = (distance >= taper.radius_km / 2) & (distance <= taper.radius_km)
    if not np.any(annulus):
        msg = "taper annulus contains no grid cell centres; use a larger radius or finer grid"
        raise DiagnosticError(msg)
    name = aliases.get("mean_sea_level_pressure", "mean_sea_level_pressure")
    baseline_rms = _annulus_rms(baseline, name, annulus, "baseline")
    if baseline_rms == 0:
        msg = "baseline pressure tendency RMS is zero in the taper annulus"
        raise DiagnosticError(msg)
    return _annulus_rms(fork, name, annulus, "fork") / baseline_rms
