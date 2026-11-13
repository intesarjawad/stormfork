from collections.abc import Mapping

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from stormfork.grid import EARTH_RADIUS_KM, Grid
from stormfork.perturb._contracts import checked_names, nonnegative_humidity, require_floating
from stormfork.perturb.base import Taper, background

MAX_DISPLACEMENT_KM = 600.0
BACKGROUND_SIGMA_DEG = 6.0
MAX_FOOTPRINT_LATITUDE_DEG = 85.0
ATMOSPHERIC_FIELDS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)


def _angular_displacement(
    grid: Grid, taper: Taper, east_km: float, north_km: float
) -> tuple[float, float]:
    dlat = float(np.rad2deg(north_km / EARTH_RADIUS_KM))
    radius_deg = float(np.rad2deg(taper.radius_km / EARTH_RADIUS_KM))
    south = min(taper.centre_lat, taper.centre_lat + dlat) - radius_deg
    north = max(taper.centre_lat, taper.centre_lat + dlat) + radius_deg
    if south < -MAX_FOOTPRINT_LATITUDE_DEG or north > MAX_FOOTPRINT_LATITUDE_DEG:
        msg = "source and destination taper footprints must stay within 85 degrees latitude"
        raise ValueError(msg)
    if south < grid.lat[0] or north > grid.lat[-1]:
        msg = "source and destination taper footprints must fit the grid latitude range"
        raise ValueError(msg)
    dlon = float(np.rad2deg(east_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(taper.centre_lat)))))
    return dlat, dlon


def _source_coordinates(grid: Grid, dlat: float, dlon: float) -> NDArray[np.float64]:
    rows, columns = np.meshgrid(
        np.arange(grid.lat.size, dtype=np.float64),
        np.arange(grid.lon.size, dtype=np.float64),
        indexing="ij",
    )
    # The extra longitude columns interpolate across the seam without wrapping latitude.
    return np.array(
        [rows - dlat / grid.spacing_deg, (columns - dlon / grid.spacing_deg) % grid.lon.size + 1]
    )


def relocate(
    ds: xr.Dataset,
    taper: Taper,
    *,
    east_km: float = 0.0,
    north_km: float = 0.0,
    names: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Translate a tapered storm anomaly by east/north distances in km, up to 600 km.

    Fields use named lat/lon dimensions on a regular grid, with any leading frame
    or level dimensions. Distances become constant angular offsets at the taper
    centre; eastward distance uses the cosine of the source centre latitude.
    Each slice uses a 6-degree Gaussian background and linear interpolation;
    longitude is periodic and shifted anomalies are zero beyond latitude edges.
    Translation preserves anomaly values in grid coordinates apart from
    interpolation. The source/destination zonal extent in km is
    cos(source latitude)/cos(destination latitude); area integrals change with
    latitude because no area rescaling is applied.
    Missing values do not spread; the original NaN mask is preserved.
    Specific humidity is clipped at zero, since moving a dry anomaly can
    otherwise leave it slightly negative. SST and static fields stay fixed.
    Eligible fields require floating-point storage, including for identity edits.
    The input is never modified; zero displacement preserves every bit and dtype.
    Optional names map WeatherNext atmospheric variable names to dataset names.

    Raises:
        KeyError: An explicitly mapped variable is missing.
        TypeError: An eligible field does not have a floating-point dtype.
        ValueError: A names key is unknown.
            Displacement is nonfinite or exceeds 600 km; either taper
            footprint extends poleward of 85 degrees or beyond the grid latitude
            range; the grid violates its contract. Footprints include centre
            latitude plus and minus radius, even for zero displacement.
    """
    distance_km = float(np.hypot(east_km, north_km))
    if not np.isfinite(distance_km) or distance_km > MAX_DISPLACEMENT_KM:
        msg = "displacement must be finite and no greater than 600 km"
        raise ValueError(msg)
    grid = Grid.from_dataset(ds)
    dlat, dlon = _angular_displacement(grid, taper, east_km, north_km)
    aliases = checked_names(ds, names, ATMOSPHERIC_FIELDS)
    require_floating(ds, aliases, ATMOSPHERIC_FIELDS)
    result = ds.copy(deep=False)
    if distance_km == 0:
        return result
    coordinates = _source_coordinates(grid, dlat, dlon)
    weights = taper.weights(grid)
    for default_name in ATMOSPHERIC_FIELDS:
        name = aliases.get(default_name, default_name)
        if name not in ds.data_vars:
            continue
        field = ds[name]
        leading_dims = [dim for dim in field.dims if dim not in ("lat", "lon")]
        ordered = field.transpose(*leading_dims, "lat", "lon")
        values = np.asarray(ordered.values, dtype=np.float64)
        smooth = background(values, sigma_grid_cells=BACKGROUND_SIGMA_DEG / grid.spacing_deg)
        residual = np.where(np.isnan(values), 0, values - smooth)
        moved = np.empty_like(values)
        for index in np.ndindex(values.shape[:-2]):
            core = np.pad(residual[index] * weights, ((0, 0), (1, 1)), mode="wrap")
            shifted = map_coordinates(
                core, coordinates, order=1, mode="constant", cval=0, prefilter=False
            )
            moved[index] = smooth[index] + shifted + residual[index] * (1 - weights)
        moved = nonnegative_humidity(default_name, moved)
        result[name] = ordered.copy(data=moved).transpose(*field.dims)
    return result
