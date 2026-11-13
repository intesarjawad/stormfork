from collections.abc import Mapping

import numpy as np
import xarray as xr

from stormfork.grid import Grid
from stormfork.perturb._contracts import checked_names, nonnegative_humidity, require_floating
from stormfork.perturb.relocate import ATMOSPHERIC_FIELDS

MAX_AMPLITUDE = 0.05
NOISE_FIELDS = (*ATMOSPHERIC_FIELDS, "sea_surface_temperature")


def noise(
    ds: xr.Dataset,
    *,
    amplitude: float,
    seed: int,
    names: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Add seeded Gaussian noise with a dimensionless amplitude in [0, 0.05].

    Atmospheric fields and SST use named lat/lon dimensions on a regular grid,
    with optional frame and level dimensions. Each frame and level receives
    independent zero-mean normal draws with standard deviation amplitude times
    its spatial population standard deviation, in the field's original units.
    Draws use canonical time, level, lat, lon order (absent dimensions skipped),
    so equivalent dimension transpositions give identical labelled results.
    Only finite cells contribute to the scale or receive noise. Specific
    humidity is clipped at zero afterwards, since noise scaled to a whole slice
    would otherwise push its driest cells negative. Constant and
    entirely missing slices stay fixed. Statics, metadata and unrelated fields
    are preserved; the input is never modified. Zero amplitude preserves every
    bit and dtype. Optional names map WeatherNext variable names to dataset names.

    Eligible fields require floating-point storage, including for identity edits.

    Raises:
        KeyError: An explicitly mapped variable is missing.
        TypeError: An eligible field does not have a floating-point dtype.
        ValueError: A names key is unknown.
            Amplitude is nonfinite or outside [0, 0.05], the seed is
            invalid for numpy.random.default_rng, or the grid is invalid.
    """
    if not np.isfinite(amplitude) or not 0 <= amplitude <= MAX_AMPLITUDE:
        msg = "amplitude must be finite and between 0 and 0.05"
        raise ValueError(msg)
    Grid.from_dataset(ds)
    rng = np.random.default_rng(seed)
    aliases = checked_names(ds, names, NOISE_FIELDS)
    require_floating(ds, aliases, NOISE_FIELDS)
    result = ds.copy(deep=False)
    if amplitude == 0:
        return result
    for default_name in NOISE_FIELDS:
        name = aliases.get(default_name, default_name)
        if name not in ds.data_vars:
            continue
        field = ds[name]
        canonical_dims = ("time", "level", "lat", "lon")
        extra_dims = sorted((dim for dim in field.dims if dim not in canonical_dims), key=str)
        ordered = field.transpose(
            *extra_dims, *(dim for dim in canonical_dims if dim in field.dims)
        )
        values = np.array(ordered.values, dtype=np.float64, copy=True)
        for index in np.ndindex(values.shape[:-2]):
            spatial = values[index]
            finite = np.isfinite(spatial)
            if not np.any(finite):
                continue
            scale = amplitude * np.std(spatial[finite])
            if scale != 0:
                spatial[finite] += rng.normal(0, scale, size=np.count_nonzero(finite))
        values = nonnegative_humidity(default_name, values)
        result[name] = ordered.copy(data=values).transpose(*field.dims)
    return result
