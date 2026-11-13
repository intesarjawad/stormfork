from collections.abc import Mapping

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.perturb._contracts import checked_names
from stormfork.perturb.base import Taper
from stormfork.perturb.relocate import ATMOSPHERIC_FIELDS

PHYSICAL_BOUNDS = (
    ("specific_humidity", 0.0, None),
    ("mean_sea_level_pressure", 87000.0, 108500.0),
    ("temperature", 170.0, 340.0),
    ("2m_temperature", 180.0, 340.0),
    ("sea_surface_temperature", 270.0, 310.0),
)
STATIC_FIELDS = ("land_sea_mask", "geopotential_at_surface")
MAX_CORE_LAND_FRACTION = 0.2
VALIDATION_FIELDS = (*ATMOSPHERIC_FIELDS, "sea_surface_temperature", *STATIC_FIELDS)


class PhysicalBoundsError(ValueError):
    """An edited field violates a chosen bound or preservation constraint.

    Arrays use named lat/lon axes in any order and optional frame/level axes.
    Error messages identify the variable and offending extreme in native units:
    pressure in Pa, temperature in K, specific humidity in kg/kg, land fraction
    dimensionless. Static geopotential uses m²/s²; preservation retains all units.
    """


def _cell_areas_km2(grid: Grid) -> NDArray[np.float64]:
    half_spacing = grid.spacing_deg / 2
    south = np.deg2rad(np.maximum(grid.lat - half_spacing, -90))
    north = np.deg2rad(np.minimum(grid.lat + half_spacing, 90))
    return np.broadcast_to(
        (EARTH_RADIUS_KM**2 * np.deg2rad(grid.spacing_deg) * (np.sin(north) - np.sin(south)))[
            :, None
        ],
        (grid.lat.size, grid.lon.size),
    )


def _core_land_fraction(ds: xr.Dataset, taper: Taper, grid: Grid, mask_name: str) -> float:
    distance = great_circle_km(taper.centre_lat, taper.centre_lon, grid.lat[:, None], grid.lon)
    core = distance <= taper.radius_km / 2
    if not np.any(core):
        msg = "core contains no grid cell centres; use a larger radius or finer grid"
        raise ValueError(msg)
    if mask_name not in ds:
        msg = f"{mask_name}: required finite land fraction in [0, 1]; extreme=missing"
        raise PhysicalBoundsError(msg)
    mask = ds[mask_name].transpose("lat", "lon").values[core]
    if not np.all(np.isfinite(mask)) or np.any((mask < 0) | (mask > 1)):
        extreme = mask[~np.isfinite(mask) | (mask < 0) | (mask > 1)][0]
        msg = f"{mask_name}: core requires finite land fraction in [0, 1]; extreme={extreme}"
        raise PhysicalBoundsError(msg)
    return float(np.average(mask, weights=_cell_areas_km2(grid)[core]))


def _check_field_bounds(
    name: str,
    field: xr.DataArray,
    lower: float,
    upper: float | None,
    undefined: xr.DataArray | None = None,
) -> None:
    values = field.values
    if undefined is not None:
        values = values[~undefined.transpose(*field.dims).values]
    nonfinite = values[~np.isfinite(values)]
    if nonfinite.size:
        msg = f"{name}: required finite values; extreme={nonfinite.flat[0]:g}"
        raise PhysicalBoundsError(msg)
    if not values.size:
        return
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < lower:
        msg = f"{name}: required >= {lower:g}; extreme={minimum:g}"
        raise PhysicalBoundsError(msg)
    if upper is not None and maximum > upper:
        msg = f"{name}: required <= {upper:g}; extreme={maximum:g}"
        raise PhysicalBoundsError(msg)


def _check_layout(original: xr.Dataset, perturbed: xr.Dataset) -> None:
    if not original.coords.to_dataset().equals(perturbed.coords.to_dataset()):
        msg = "original and perturbed coordinates must match"
        raise ValueError(msg)
    if set(original.data_vars) != set(perturbed.data_vars):
        msg = "original and perturbed must contain the same variables"
        raise ValueError(msg)
    for name in original.data_vars:
        if set(original[name].dims) != set(perturbed[name].dims):
            msg = f"{name}: original and perturbed dimensions must match"
            raise ValueError(msg)


def _check_preservation(
    original: xr.Dataset, perturbed: xr.Dataset, aliases: Mapping[str, str]
) -> None:
    _check_layout(original, perturbed)
    for name in original.data_vars:
        before = original[name].values
        after = perturbed[name].transpose(*original[name].dims).values
        new_nonfinite = ~np.isfinite(after) & np.isfinite(before)
        if bool(new_nonfinite.any()):
            extreme = after[new_nonfinite].flat[0]
            msg = f"{name}: no new nonfinite values; extreme={extreme:g}"
            raise PhysicalBoundsError(msg)
    for default in STATIC_FIELDS:
        name = aliases.get(default, default)
        if name in original and not original[name].equals(
            perturbed[name].transpose(*original[name].dims)
        ):
            difference = abs(perturbed[name] - original[name])
            msg = f"{name}: required unchanged; extreme absolute change={float(difference.max()):g}"
            raise PhysicalBoundsError(msg)


def validate(
    original: xr.Dataset,
    perturbed: xr.Dataset,
    *,
    moved_taper: Taper | None = None,
    names: Mapping[str, str] | None = None,
) -> float | None:
    """Check edited arrays on named lat/lon axes with optional frame/level axes.

    Pressure bounds use Pa, temperatures K, and specific humidity kg/kg. Bounded
    fields must be finite everywhere, except that sea surface temperature may be
    NaN in any cell with land (land_sea_mask above zero): analyses leave it
    undefined over land, including coastal cells that are mostly sea. Other
    fields may contain nonfinite values only where the original was nonfinite.
    Land mask and surface geopotential must retain their values. For relocation,
    pass the destination taper: its inner half must have an area-weighted mean
    land fraction strictly below 0.2.
    Optional names map WeatherNext variable names to dataset names.

    Returns:
        The relocated core's area-weighted land fraction when moved_taper is
        given, else None.

    Raises:
        PhysicalBoundsError: A field violates a bound or preservation constraint.
        KeyError: An explicitly mapped variable is missing from either dataset.
        ValueError: A names key is unknown, grids or variables differ, or the core
            has no grid cell centres.
    """
    grid = Grid.from_dataset(perturbed)
    aliases = checked_names(original, names, VALIDATION_FIELDS)
    checked_names(perturbed, names, VALIDATION_FIELDS)
    _check_preservation(original, perturbed, aliases)
    mask_name = aliases.get("land_sea_mask", "land_sea_mask")
    for default, lower, upper in PHYSICAL_BOUNDS:
        name = aliases.get(default, default)
        if name not in perturbed:
            continue
        undefined = (
            perturbed[name].isnull() & (perturbed[mask_name] > 0)
            if default == "sea_surface_temperature" and mask_name in perturbed
            else None
        )
        _check_field_bounds(name, perturbed[name], lower, upper, undefined)
    if moved_taper is None:
        return None
    fraction = _core_land_fraction(perturbed, moved_taper, grid, mask_name)
    if fraction >= MAX_CORE_LAND_FRACTION:
        msg = f"{mask_name}: core land fraction must be < 0.2; extreme={fraction:g}"
        raise PhysicalBoundsError(msg)
    return fraction


def dry_run_statistics(
    original: xr.Dataset,
    perturbed: xr.Dataset,
    taper: Taper,
    *,
    names: Mapping[str, str] | None = None,
) -> str:
    """Format edited-minus-original min/max/mean in each field's native units.

    Arrays use named lat/lon axes and optional frame/level axes. Reductions cover
    all finite changes. Entirely missing changes report nan. Taper area sums
    spherical cell areas whose centres lie inside the radius, in km². Core land
    fraction averages the mask by cell area inside radius/2. Pass the destination
    taper for relocation. This function returns text without performing I/O.
    """
    grid = Grid.from_dataset(perturbed)
    aliases = checked_names(original, names, VALIDATION_FIELDS)
    checked_names(perturbed, names, VALIDATION_FIELDS)
    _check_layout(original, perturbed)
    lines: list[str] = []
    for name in perturbed.data_vars:
        change = np.asarray((perturbed[name] - original[name]).values, dtype=np.float64)
        finite = change[np.isfinite(change)]
        minimum, maximum, mean = (
            (float(finite.min()), float(finite.max()), float(finite.mean()))
            if finite.size
            else (np.nan, np.nan, np.nan)
        )
        lines.append(f"{name}: min={minimum:g} max={maximum:g} mean={mean:g}")
    distance = great_circle_km(taper.centre_lat, taper.centre_lon, grid.lat[:, None], grid.lon)
    area = float(_cell_areas_km2(grid)[distance <= taper.radius_km].sum())
    mask_name = aliases.get("land_sea_mask", "land_sea_mask")
    fraction = _core_land_fraction(perturbed, taper, grid, mask_name)
    lines.extend((f"taper area: {area:g} km²", f"core land fraction: {fraction:g}"))
    return "\n".join(lines)
