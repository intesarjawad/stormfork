from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def checked_names(
    ds: xr.Dataset, names: Mapping[str, str] | None, supported: Sequence[str]
) -> Mapping[str, str]:
    """Reject unknown aliases and explicitly requested variables that are absent."""
    aliases = names or {}
    unknown = aliases.keys() - set(supported)
    if unknown:
        msg = f"unknown names keys: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    for default, name in aliases.items():
        if name not in ds.data_vars:
            msg = f"{default}: mapped variable {name!r} is missing from the dataset"
            raise KeyError(msg)
    return aliases


def require_floating(ds: xr.Dataset, aliases: Mapping[str, str], fields: Sequence[str]) -> None:
    """Require floating-point storage for fields eligible for an edit."""
    for default in fields:
        name = aliases.get(default, default)
        if name in ds.data_vars and not np.issubdtype(ds[name].dtype, np.floating):
            msg = f"{name}: perturbations require a floating-point field; got {ds[name].dtype}"
            raise TypeError(msg)


def nonnegative_humidity(default_name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip edited specific humidity at zero; other fields pass through unchanged.

    Humidity is a mass fraction, but the edits add signed anomalies, so a nearly
    dry cell can end slightly below zero. Clipping changes only those cells and
    leaves the edit everywhere else as computed; NaN stays NaN.
    """
    return np.maximum(values, 0) if default_name == "specific_humidity" else values
