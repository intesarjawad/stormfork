from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

LAT = np.arange(-90, 91, 1.0)
LON = np.arange(0, 360, 1.0)
LEVELS_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
STEP = timedelta(hours=6)
EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius.
SPACING_TOLERANCE_DEG = 1e-6
MIN_AXIS_POINTS = 2  # A spacing requires at least one pair of coordinates.


def _axis_spacing(axis: NDArray[np.float64], name: str) -> float:
    if axis.ndim != 1 or axis.size < MIN_AXIS_POINTS or not np.all(np.isfinite(axis)):
        msg = f"{name} must be a finite 1-D coordinate with at least two points"
        raise ValueError(msg)
    differences = np.diff(axis)
    spacing = float(differences[0])
    if np.any(differences <= 0) or not np.allclose(
        differences, spacing, atol=SPACING_TOLERANCE_DEG, rtol=0
    ):
        msg = f"{name} must be ascending with constant spacing within 1e-6 degrees"
        raise ValueError(msg)
    return spacing


@dataclass(frozen=True)
class Grid:
    """Regular latitude-longitude coordinates and their shared spacing in degrees."""

    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    spacing_deg: float

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "Grid":
        """Read ascending, equally spaced axes with global 0-based longitudes.

        Raises:
            ValueError: A coordinate is missing or violates the grid contract.
        """
        for name in ("lat", "lon"):
            if name not in ds.coords:
                msg = f"Dataset is missing the {name} coordinate"
                raise ValueError(msg)
        lat = np.asarray(ds.coords["lat"].values, dtype=np.float64)
        lon = np.asarray(ds.coords["lon"].values, dtype=np.float64)
        spacing = _axis_spacing(lat, "lat")
        lon_spacing = _axis_spacing(lon, "lon")
        if not np.isclose(spacing, lon_spacing, atol=SPACING_TOLERANCE_DEG, rtol=0):
            msg = f"lat spacing {spacing} and lon spacing {lon_spacing} must match"
            raise ValueError(msg)
        if not np.allclose(
            [lon[0], lon[-1]], [0, 360 - spacing], atol=SPACING_TOLERANCE_DEG, rtol=0
        ):
            msg = f"lon must cover 0 to {360 - spacing} degrees without a duplicate endpoint"
            raise ValueError(msg)
        return cls(lat, lon, spacing)


GRID_1DEG = Grid(LAT, LON, 1.0)


def great_circle_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Return haversine distances in km for broadcastable coordinates in degrees.

    Scalar coordinates return a zero-dimensional array.
    """
    phi1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    phi2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    delta_lon = np.deg2rad(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    haversine = np.sin((phi2 - phi1) / 2) ** 2 + (
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lon / 2) ** 2
    )
    # Roundoff near antipodes can put the haversine just outside [0, 1].
    angle = 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return np.asarray(EARTH_RADIUS_KM * angle, dtype=np.float64)
