"""Turn a run's model input and forecast into the fields its atlases encode.

The atlases carry member 0's fields; every member's track is exported
separately, so the page never shows one member's map as the forecast.
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

ATLAS_VARIABLES = {
    "mslp": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "z500": "geopotential",
    "tp6h": "total_precipitation_6hr",
}
Z500_LEVEL_HPA = 500


def _member_zero_frames(dataset: xr.Dataset, name: str) -> NDArray[np.float64]:
    if "level" in dataset[name].dims:
        dataset = dataset.sel(level=Z500_LEVEL_HPA)
    field = dataset[name]
    selection: dict[str, int] = {dim: 0 for dim in ("member", "batch") if dim in field.dims}
    selected = field.isel(selection).transpose("time", "lat", "lon")
    return np.asarray(selected.values, dtype=np.float64)


def atlas_fields(inputs: xr.Dataset, forecast: xr.Dataset) -> dict[str, NDArray[np.float64]]:
    """Return the six atlas fields of member 0, frame 0 being the state at init.

    inputs is the model input the run started from, its last time the init;
    forecast holds the six-hour steps in order, with optional member and batch
    axes. Each field comes back as (steps + 1, lat, lon) in native units.
    Precipitation accumulates over a step, so it has no value at the init and
    its frame 0 is zero.

    Raises:
        KeyError: A variable or the 500 hPa level is missing.
    """
    fields: dict[str, NDArray[np.float64]] = {}
    for file, name in ATLAS_VARIABLES.items():
        steps = _member_zero_frames(forecast, name)
        start = np.zeros_like(steps[0]) if file == "tp6h" else _member_zero_frames(inputs, name)[-1]
        fields[file] = np.concatenate([start[None], steps])
    return fields
