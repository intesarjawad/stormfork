from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from http import HTTPStatus
from pathlib import Path
from time import sleep
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from stormfork.downloads import open_https
from stormfork.errors import UserError
from stormfork.grid import LEVELS_HPA, STEP
from stormfork.ic.era5 import ATMOSPHERE, SURFACE, build_input
from stormfork.ic.layout import POINTS_PER_DEGREE, QUARTER_DEGREE, SampleStatics, coarsen
from stormfork.scores.fetch import HTTP_TIMEOUT_SECONDS, RETRY_DELAYS_SECONDS, request_bytes

Method = Literal["subsample", "cell_mean"]

METHODS: tuple[Method, ...] = ("subsample", "cell_mean")
# The AWS mirror answers archive requests with throttling errors, so the Google bucket is used.
IFS_OPEN_ROOT = "https://storage.googleapis.com/ecmwf-open-data"
STANDARD_GRAVITY = 9.80665  # m s-2; converts open-data geopotential height to geopotential.
# GRIB short names, keyed to the ARCO variable each one replaces.
PRESSURE_PARAMS = {
    "t": "temperature",
    "gh": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}
SURFACE_PARAMS = {
    "2t": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
}
HYBRID_MAX_RATIO = 0.25
HYBRID_FROM = date(2023, 7, 12)
# The fields whose values depend most on how the one-degree grid was made.
COARSENING_FIELDS = ("vertical_velocity", "specific_humidity", "mean_sea_level_pressure")
INPUT_FRAMES = 2
QUARTER_LAT = np.arange(-90, 90 + QUARTER_DEGREE, QUARTER_DEGREE)
QUARTER_LON = np.arange(0, 360, QUARTER_DEGREE)


class IndexEntry(BaseModel):
    """One message of an open-data GRIB file, from its sidecar index."""

    model_config = ConfigDict(extra="ignore")

    param: str
    levtype: str
    levelist: str | None = None
    offset: int = Field(alias="_offset")
    length: int = Field(alias="_length")


def ifs_url(valid: datetime, suffix: Literal["grib2", "index"] = "grib2") -> str:
    """Return the step-0 file of the archived 0.25-degree IFS analysis cycle.

    The 00 and 12 UTC cycles are the `oper` stream; 06 and 18 UTC are `scda`.

    Raises:
        ValueError: The time is not on a six-hour UTC cycle.
    """
    utc = valid.replace(tzinfo=UTC) if valid.tzinfo is None else valid.astimezone(UTC)
    if utc.hour % 6 or utc.minute or utc.second or utc.microsecond:
        msg = "IFS open data are issued on six-hour UTC cycles"
        raise ValueError(msg)
    stream = "oper" if utc.hour % 12 == 0 else "scda"
    name = f"{utc:%Y%m%d%H}0000-0h-{stream}-fc.{suffix}"
    return f"{IFS_OPEN_ROOT}/{utc:%Y%m%d}/{utc:%H}z/ifs/0p25/{stream}/{name}"


def select_ranges(index_text: str) -> list[tuple[int, int]]:
    """Return merged (offset, length) byte ranges of the messages the builder needs.

    Raises:
        ValueError: The index lacks a required field and level.
    """
    wanted = {("pl", short, level) for short in PRESSURE_PARAMS for level in LEVELS_HPA} | {
        ("sfc", short, None) for short in SURFACE_PARAMS
    }
    found: dict[tuple[str, str, int | None], IndexEntry] = {}
    for line in index_text.splitlines():
        entry = IndexEntry.model_validate_json(line)
        level = None if entry.levelist is None else int(entry.levelist)
        if (entry.levtype, entry.param, level) in wanted:
            found[entry.levtype, entry.param, level] = entry
    missing = sorted(wanted - found.keys(), key=str)
    if missing:
        msg = f"IFS open-data index lacks {len(missing)} required messages, first {missing[0]}"
        raise ValueError(msg)
    ranges: list[tuple[int, int]] = []
    for entry in sorted(found.values(), key=lambda item: item.offset):
        if ranges and ranges[-1][0] + ranges[-1][1] == entry.offset:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + entry.length)
        else:
            ranges.append((entry.offset, entry.length))
    return ranges


def _request_range(url: str, offset: int, length: int) -> bytes:
    if urlsplit(url).scheme != "https":
        msg = "public data requests require HTTPS"
        raise ValueError(msg)
    request = Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})  # noqa: S310
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            with open_https(request, HTTP_TIMEOUT_SECONDS) as response:
                body = response.read()
        except (URLError, TimeoutError) as error:
            if (
                isinstance(error, HTTPError)
                and error.code < HTTPStatus.INTERNAL_SERVER_ERROR
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
            ) or attempt == len(RETRY_DELAYS_SECONDS):
                msg = f"could not reach {url} ({error})"
                raise UserError(msg, "retry the same command", 3) from error
            sleep(RETRY_DELAYS_SECONDS[attempt])
        else:
            if len(body) != length:
                msg = f"{url} returned {len(body)} bytes for a {length}-byte range"
                raise ValueError(msg)
            return body
    msg = "HTTP retry loop exhausted"
    raise RuntimeError(msg)


def download_messages(valid: datetime, directory: Path) -> Path:
    """Write only the needed messages of one cycle to a local GRIB file."""
    index = request_bytes(ifs_url(valid, "index")).decode()
    path = directory / f"ifs_open_{valid:%Y%m%d%H}.grib2"
    with path.open("wb") as file:
        for offset, length in select_ranges(index):
            file.write(_request_range(ifs_url(valid), offset, length))
    return path


def decode_messages(path: Path) -> dict[str, xr.DataArray]:
    """Read the needed fields of a downloaded file, keyed by GRIB short name."""
    fields: dict[str, xr.DataArray] = {}

    def read(**keys: str) -> xr.Dataset:
        return xr.open_dataset(
            path, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": keys}
        )

    with read(typeOfLevel="isobaricInhPa") as pressure:
        for short in PRESSURE_PARAMS:
            fields[short] = _detach(pressure[short])
    for short in SURFACE_PARAMS:
        with read(shortName=short) as surface:
            (variable,) = surface.data_vars
            fields[short] = _detach(surface[variable])
    return fields


def _detach(field: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(field.values),
        dims=field.dims,
        coords={dim: np.asarray(field[dim].values) for dim in field.dims},
    )


def to_store(
    frames: Sequence[Mapping[str, xr.DataArray]], times: Sequence[np.datetime64]
) -> xr.Dataset:
    """Give decoded fields the ARCO names, units and axes the ERA5 builder reads.

    Geopotential height becomes geopotential and longitudes run 0 to 360.

    Raises:
        ValueError: A field is not on the expected axes or the grids differ.
    """
    first = frames[0]["t"]
    latitude = np.asarray(first["latitude"].values)
    longitude = np.asarray(first["longitude"].values) % 360
    longitude_order = np.argsort(longitude)
    data_vars: dict[str, xr.DataArray] = {}
    for params, axes in (
        (PRESSURE_PARAMS, ("isobaricInhPa", "latitude", "longitude")),
        (SURFACE_PARAMS, ("latitude", "longitude")),
    ):
        for short, name in params.items():
            stacked: list[np.ndarray] = []
            for frame in frames:
                field = frame[short]
                if field.dims != axes or not (
                    np.array_equal(field["latitude"].values, latitude)
                    and np.array_equal(np.asarray(field["longitude"].values) % 360, longitude)
                ):
                    msg = f"{short} must be on the {axes} axes of one shared grid"
                    raise ValueError(msg)
                values = np.asarray(field.values)[..., longitude_order]
                stacked.append(values * STANDARD_GRAVITY if short == "gh" else values)
            dims = ("time", *("level" if axis == "isobaricInhPa" else axis for axis in axes))
            data_vars[name] = xr.DataArray(np.stack(stacked), dims=dims)
    return xr.Dataset(
        data_vars,
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "level": np.asarray(first["isobaricInhPa"].values),
            "latitude": latitude,
            "longitude": longitude[longitude_order],
        },
    )


def fetch_ifs_store(init: datetime, directory: Path) -> xr.Dataset:
    """Download and decode the two input frames, step 0 of the cycles at init - 6 h and init."""
    utc_init = init.replace(tzinfo=UTC) if init.tzinfo is None else init.astimezone(UTC)
    valids = [utc_init + offset * STEP for offset in (-1, 0)]
    frames = [decode_messages(download_messages(valid, directory)) for valid in valids]
    return to_store(frames, [np.datetime64(valid.replace(tzinfo=None), "ns") for valid in valids])


def hybrid_store(ifs: xr.Dataset, era5: xr.Dataset) -> xr.Dataset:
    """Add ERA5 sea surface temperature on the IFS grid and frames.

    Raises:
        KeyError: ERA5 lacks a frame or grid point of the IFS store.
    """
    sst = era5[["sea_surface_temperature"]].sel(
        time=ifs["time"].values,
        latitude=ifs["latitude"].values,
        longitude=ifs["longitude"].values,
    )
    return ifs.assign(sea_surface_temperature=sst["sea_surface_temperature"])


def build_with_method(
    init: datetime,
    store: xr.Dataset,
    num_steps: int = 40,
    *,
    statics: SampleStatics,
    method: Method,
) -> xr.Dataset:
    """Build input frames, reducing the quarter-degree grid with the chosen method.

    The ERA5 builder always subsamples, so other reductions are computed first and
    carried on the quarter-degree points it keeps.
    """
    if method == "subsample":
        return build_input(init, store, num_steps, statics=statics)
    utc_init = init.replace(tzinfo=UTC) if init.tzinfo is None else init.astimezone(UTC)
    frames = np.datetime64(utc_init.replace(tzinfo=None), "ns") + np.array(
        [-1, 0]
    ) * np.timedelta64(STEP, "ns")
    window = (
        store[list(ATMOSPHERE + SURFACE)]
        .sel(time=frames, level=list(LEVELS_HPA))
        .rename(latitude="lat", longitude="lon")
    )
    reduced = coarsen(window, method)
    carried: dict[str, xr.DataArray] = {}
    for name in ATMOSPHERE + SURFACE:
        values = np.asarray(reduced[name].values, dtype=np.float32)
        carrier = np.full(
            (*values.shape[:-2], QUARTER_LAT.size, QUARTER_LON.size), np.nan, dtype=np.float32
        )
        carrier[..., ::POINTS_PER_DEGREE, ::POINTS_PER_DEGREE] = values
        carried[name] = xr.DataArray(carrier, dims=reduced[name].dims)
    carrier_store = xr.Dataset(
        carried,
        coords={
            "time": frames,
            "level": np.asarray(LEVELS_HPA),
            "lat": QUARTER_LAT,
            "lon": QUARTER_LON,
        },
    ).rename(lat="latitude", lon="longitude")
    return build_input(init, carrier_store, num_steps, statics=statics)


def build_hybrid(
    init: datetime, ifs: xr.Dataset, era5: xr.Dataset, *, statics: SampleStatics, method: Method
) -> xr.Dataset:
    """Build input frames from IFS open-data atmosphere and ERA5 sea surface temperature."""
    return build_with_method(init, hybrid_store(ifs, era5), statics=statics, method=method)


def _input_values(dataset: xr.Dataset, name: str) -> np.ndarray:
    return dataset[name].isel(time=slice(0, INPUT_FRAMES)).values.astype(np.float64)


def rms_difference(actual: xr.Dataset, sample: xr.Dataset, name: str) -> float:
    """Return the RMS difference over both input frames, all levels and finite paired cells.

    Raises:
        ValueError: No cell is finite in both datasets.
    """
    difference = _input_values(actual, name) - _input_values(sample, name)
    finite = difference[np.isfinite(difference)]
    if not finite.size:
        msg = f"{name} has no finite paired cells"
        raise ValueError(msg)
    return float(np.sqrt(np.mean(finite**2)))


def sample_std(sample: xr.Dataset, name: str) -> float:
    """Return the unweighted standard deviation of the sample's input frames, finite cells only."""
    values = _input_values(sample, name)
    return float(np.std(values[np.isfinite(values)]))


def coarsening_score(actual: xr.Dataset, sample: xr.Dataset) -> float:
    """Mean over the grid-sensitive fields of RMS difference divided by sample deviation."""
    return float(
        np.mean(
            [
                rms_difference(actual, sample, name) / sample_std(sample, name)
                for name in COARSENING_FIELDS
            ]
        )
    )


def best_coarsening(scores: Mapping[Method, float]) -> Method:
    """Return the method with the lower score; a tie keeps subsampling."""
    return min(METHODS, key=lambda method: scores[method])


def hybrid_is_source(hybrid: Mapping[str, float], era5: Mapping[str, float]) -> bool:
    """Apply the rule: every atmospheric variable at most a quarter of the ERA5 RMS difference."""
    return all(hybrid[name] <= HYBRID_MAX_RATIO * era5[name] for name in ATMOSPHERE)


def decision_line(hybrid: Mapping[str, float], era5: Mapping[str, float]) -> str:
    """Describe the source decision with the variable closest to or beyond the limit."""
    ratios = {name: hybrid[name] / era5[name] for name in ATMOSPHERE}
    worst = max(ratios, key=lambda name: ratios[name])
    if hybrid_is_source(hybrid, era5):
        return (
            "IC decision: hybrid (IFS open data atmosphere, ERA5 SST) is the source for hindcast "
            f"inits from {HYBRID_FROM.isoformat()} onward; the largest RMS ratio to ERA5 is "
            f"{ratios[worst]:.3f} ({worst}), limit {HYBRID_MAX_RATIO}."
        )
    return (
        "IC decision: ERA5 is the source for all hindcast inits; "
        f"{worst} has a hybrid RMS of {ratios[worst]:.3f} of ERA5's, limit {HYBRID_MAX_RATIO}."
    )


def coarsening_line(scores: Mapping[Method, float]) -> str:
    """Describe the chosen coarsening method and the scores behind it."""
    best = best_coarsening(scores)
    detail = ", ".join(f"{method} {scores[method]:.4f}" for method in METHODS)
    return f"Coarsening: {best} (mean normalised RMS difference: {detail}); both builders use it."
