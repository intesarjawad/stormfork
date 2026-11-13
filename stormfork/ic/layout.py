from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import xarray as xr

from stormfork.grid import LAT, LON

STATIC_NAMES = ("geopotential_at_surface", "land_sea_mask")
QUARTER_DEGREE = 0.25
POINTS_PER_DEGREE = 4


@dataclass(frozen=True)
class SampleStatics:
    """Sample terrain and land fraction on the model's one-degree grid."""

    geopotential_at_surface: xr.DataArray
    land_sea_mask: xr.DataArray

    def __post_init__(self) -> None:
        for name, field in zip(
            STATIC_NAMES, (self.geopotential_at_surface, self.land_sea_mask), strict=True
        ):
            if (
                field.dims != ("lat", "lon")
                or field.dtype != np.dtype("float32")
                or not np.array_equal(field["lat"].values, LAT)
                or not np.array_equal(field["lon"].values, LON)
            ):
                msg = f"{name} must be float32 on the ascending global one-degree lat/lon grid"
                raise ValueError(msg)


def load_sample_statics(path: str | PathLike[str]) -> SampleStatics:
    """Load only terrain and land fraction, closing the sample before returning."""
    with xr.open_dataset(path, engine="h5netcdf") as sample:
        return SampleStatics(
            *(sample[name].copy(data=np.asarray(sample[name].values)) for name in STATIC_NAMES)
        )


def coarsen(
    dataset: xr.Dataset, method: Literal["subsample", "cell_mean"] = "subsample"
) -> xr.Dataset:
    """Reduce a global quarter-degree grid to the model's ascending one-degree grid.

    Cell means use centred five-point trapezoidal weights on each axis,
    cosine latitude area weights, periodic longitude and clipped polar caps.
    Missing points are excluded from the weighted mean.

    Raises:
        ValueError: Coordinates or the reduction method violate the grid contract.
    """
    ordered = dataset.sortby("lat")
    if not np.array_equal(
        ordered.lat, np.arange(-90, 90 + QUARTER_DEGREE, QUARTER_DEGREE)
    ) or not np.array_equal(ordered.lon, np.arange(0, 360, QUARTER_DEGREE)):
        msg = "Coarsening requires a global quarter-degree lat/lon grid with 0-based longitude"
        raise ValueError(msg)
    if method == "subsample":
        return ordered.isel(
            lat=slice(None, None, POINTS_PER_DEGREE), lon=slice(None, None, POINTS_PER_DEGREE)
        )
    if method != "cell_mean":
        msg = f"Unknown coarsening method: {method}"
        raise ValueError(msg)
    # The outer points are shared by adjacent cells, hence their half weights.
    offsets = np.arange(-2, 3)
    edge_weights = np.array([0.5, 1, 1, 1, 0.5])
    latitude_indices = np.arange(0, ordered.sizes["lat"], POINTS_PER_DEGREE)[:, None] + offsets
    longitude_indices = (
        np.arange(0, ordered.sizes["lon"], POINTS_PER_DEGREE)[:, None] + offsets
    ) % ordered.sizes["lon"]
    latitude_weights = np.cos(np.deg2rad(-90 + latitude_indices * QUARTER_DEGREE)) * edge_weights
    latitude_weights[(latitude_indices < 0) | (latitude_indices >= ordered.sizes["lat"])] = 0
    latitude_indices = latitude_indices.clip(0, ordered.sizes["lat"] - 1)
    lat_index = xr.DataArray(latitude_indices, dims=("lat", "lat_window"))
    lon_index = xr.DataArray(longitude_indices, dims=("lon", "lon_window"))
    weights = xr.DataArray(latitude_weights, dims=("lat", "lat_window")) * xr.DataArray(
        edge_weights, dims="lon_window"
    )
    windows = ordered.isel(lat=lat_index, lon=lon_index).drop_vars(["lat", "lon"])
    result = windows.weighted(weights).mean(("lat_window", "lon_window"))
    result.coords["lat"] = LAT.astype(np.float32)
    result.coords["lon"] = LON.astype(np.float32)
    return result
