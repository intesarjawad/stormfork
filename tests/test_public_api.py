import numpy as np
import pytest
import xarray as xr

from stormfork import perturb
from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km


def test_public_exports():
    assert perturb.__all__ == [
        "PhysicalBoundsError",
        "Taper",
        "adjustment_ratio",
        "noise",
        "relocate",
        "scale_intensity",
        "sst_anomaly",
        "validate",
    ]
    for name in perturb.__all__:
        assert callable(getattr(perturb, name))


@pytest.mark.parametrize("spacing", [1.0, 0.25])
def test_public_relocate_on_regular_grid(spacing: float):
    centre_lat, centre_lon = 25.0, 280.0
    ds = xr.Dataset(
        coords={
            "lat": np.arange(10, 40 + spacing, spacing),
            "lon": np.arange(0, 360, spacing),
        }
    )
    grid = Grid.from_dataset(ds)
    distance = great_circle_km(centre_lat, centre_lon, grid.lat[:, None], grid.lon)
    ds["mslp"] = (("lat", "lon"), 101000 - 6000 * np.exp(-0.5 * (distance / 130) ** 2))
    original = ds.copy(deep=True)
    east_km, north_km = 180.0, 240.0
    moved = perturb.relocate(
        ds,
        perturb.Taper(centre_lat, centre_lon, 900),
        east_km=east_km,
        north_km=north_km,
        names={"mean_sea_level_pressure": "mslp"},
    )
    expected_lat = centre_lat + np.rad2deg(north_km / EARTH_RADIUS_KM)
    expected_lon = centre_lon + np.rad2deg(
        east_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(centre_lat)))
    )
    row, column = np.unravel_index(np.argmin(moved.mslp.values), moved.mslp.shape)
    assert float(moved.lat[row]) == pytest.approx(expected_lat, abs=spacing)
    assert float(moved.lon[column]) == pytest.approx(expected_lon, abs=spacing)
    assert ds.identical(original)
