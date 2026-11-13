from collections.abc import Mapping

import numpy as np
import xarray as xr

from stormfork.grid import Grid
from stormfork.perturb._contracts import checked_names, nonnegative_humidity, require_floating
from stormfork.perturb.base import Taper, background
from stormfork.perturb.relocate import ATMOSPHERIC_FIELDS, BACKGROUND_SIGMA_DEG

MIN_FACTOR = 0.5
MAX_FACTOR = 1.5


def scale_intensity(
    ds: xr.Dataset,
    taper: Taper,
    *,
    factor: float,
    names: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Scale a tapered atmospheric anomaly by a dimensionless factor in [0.5, 1.5].

    Fields use named lat/lon dimensions on a regular grid, with any leading frame
    or level dimensions. Each slice uses a 6-degree Gaussian background; values
    retain their original units. Smoothing ignores missing values and the output
    retains exactly the original NaN mask. Specific humidity is clipped at zero,
    since strengthening a dry anomaly can otherwise leave it slightly negative.
    SST and static fields stay fixed. The input is never modified; factor 1
    preserves every bit and dtype. Optional names map WeatherNext atmospheric
    variable names to dataset names.

    Eligible fields require floating-point storage, including for identity edits.

    Raises:
        KeyError: An explicitly mapped variable is missing.
        TypeError: An eligible field does not have a floating-point dtype.
        ValueError: A names key is unknown.
            Factor is nonfinite or outside [0.5, 1.5], or the grid
            violates its contract.
    """
    if not np.isfinite(factor) or not MIN_FACTOR <= factor <= MAX_FACTOR:
        msg = "factor must be finite and between 0.5 and 1.5"
        raise ValueError(msg)
    grid = Grid.from_dataset(ds)
    aliases = checked_names(ds, names, ATMOSPHERIC_FIELDS)
    require_floating(ds, aliases, ATMOSPHERIC_FIELDS)
    result = ds.copy(deep=False)
    if factor == 1.0:
        return result
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
        scaled = smooth + (values - smooth) * (1 + (factor - 1) * weights)
        scaled = nonnegative_humidity(default_name, scaled)
        result[name] = ordered.copy(data=scaled).transpose(*field.dims)
    return result
