from collections.abc import Mapping

import numpy as np
import xarray as xr

from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.perturb._contracts import checked_names, require_floating

MAX_DELTA_K = 3.0
POLE_LATITUDE_DEG = 90.0
OCEAN_MASK_THRESHOLD = 0.5
AIR_RESPONSE_FRACTION = 0.5
AIR_RESPONSE_LEVELS_HPA = (1000, 925)
TEMPERATURE_FIELDS = ("sea_surface_temperature", "2m_temperature", "temperature")


def _check_geometry(grid: Grid, centre: tuple[float, float], sigma_km: float) -> None:
    centre_lat, centre_lon = centre
    if not np.isfinite(sigma_km) or sigma_km <= 0:
        msg = "sigma_km must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(centre_lat) or not -POLE_LATITUDE_DEG <= centre_lat <= POLE_LATITUDE_DEG:
        msg = "centre_lat must be finite and between -90 and 90 degrees"
        raise ValueError(msg)
    if not np.isfinite(centre_lon):
        msg = "centre_lon must be finite, in degrees"
        raise ValueError(msg)
    spacing_km = EARTH_RADIUS_KM * np.deg2rad(grid.spacing_deg)
    if sigma_km < spacing_km:
        msg = f"sigma_km must be at least one grid spacing ({spacing_km:g} km)"
        raise ValueError(msg)


def _check_required_fields(ds: xr.Dataset, aliases: Mapping[str, str]) -> None:
    for default in ("sea_surface_temperature", "land_sea_mask"):
        if aliases.get(default, default) not in ds.data_vars:
            raise KeyError(aliases.get(default, default))
    temperature_name = aliases.get("temperature", "temperature")
    if temperature_name in ds.data_vars:
        if "level" not in ds[temperature_name].dims or "level" not in ds.coords:
            msg = f"{temperature_name}: required pressure levels {AIR_RESPONSE_LEVELS_HPA} hPa"
            raise KeyError(msg)
        missing = set(AIR_RESPONSE_LEVELS_HPA) - set(ds.level.values)
        if missing:
            msg = f"{temperature_name}: missing required pressure levels {sorted(missing)} hPa"
            raise KeyError(msg)


def sst_anomaly(
    ds: xr.Dataset,
    *,
    centre: tuple[float, float],
    sigma_km: float,
    delta_k: float,
    names: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Add a Gaussian SST anomaly in kelvin, restricted to ocean cells.

    The centre is (latitude, longitude) in degrees; sigma_km is the great-circle
    standard deviation and must span at least one meridional grid spacing in km.
    The analytic Gaussian is sampled at grid nodes without amplitude rescaling:
    its peak equals delta_k only when the centre is an ocean node.
    Fields use named lat/lon dimensions on a regular grid, with optional frame
    dimensions and pressure levels in hPa on level. Ocean is land_sea_mask < 0.5.
    The 2m_temperature and temperature at 1000/925 hPa receive half the anomaly
    when present. Land and other levels retain every bit and dtype; the input is
    never modified. Optional names map WeatherNext variable names to dataset names.

    Raises:
        TypeError: A temperature field does not have a floating-point dtype.
        ValueError: A names key is unknown, delta is nonfinite or outside [-3, 3] K,
            sigma is smaller than one grid spacing in km, geometry is invalid,
            or the grid violates its contract.
        KeyError: SST, land mask, an explicitly mapped variable, or a required
            1000/925 hPa level of an included temperature field is missing.
    """
    if not np.isfinite(delta_k) or not -MAX_DELTA_K <= delta_k <= MAX_DELTA_K:
        msg = "delta_k must be finite and between -3 and 3 K"
        raise ValueError(msg)
    grid = Grid.from_dataset(ds)
    _check_geometry(grid, centre, sigma_km)
    aliases = checked_names(ds, names, (*TEMPERATURE_FIELDS, "land_sea_mask"))
    require_floating(ds, aliases, TEMPERATURE_FIELDS)
    _check_required_fields(ds, aliases)
    result = ds.copy(deep=False)
    if delta_k == 0:
        return result
    distance_km = great_circle_km(*centre, grid.lat[:, None], grid.lon)
    change = xr.DataArray(
        delta_k * np.exp(-0.5 * (distance_km / sigma_km) ** 2),
        dims=("lat", "lon"),
        coords={"lat": ds.lat, "lon": ds.lon},
    )
    mask_name = aliases.get("land_sea_mask", "land_sea_mask")
    ocean = ds[mask_name] < OCEAN_MASK_THRESHOLD
    for default_name in TEMPERATURE_FIELDS:
        name = aliases.get(default_name, default_name)
        if default_name != "sea_surface_temperature" and name not in ds.data_vars:
            continue
        field = ds[name]
        selected = ocean
        increment = change
        if default_name != "sea_surface_temperature":
            increment = change * AIR_RESPONSE_FRACTION
        if default_name == "temperature":
            selected = selected & ds["level"].isin(AIR_RESPONSE_LEVELS_HPA)
        selected = selected.broadcast_like(field).transpose(*field.dims)
        increment = increment.broadcast_like(field).transpose(*field.dims)
        values = field.values.copy()
        # Mask the addition itself to preserve signed zeros and NaN bits over land.
        np.add(values, increment.values, out=values, where=selected.values)
        result[name] = field.copy(data=values)
    return result
