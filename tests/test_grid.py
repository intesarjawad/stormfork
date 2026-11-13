from dataclasses import FrozenInstanceError
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from stormfork.grid import GRID_1DEG, LAT, LEVELS_HPA, LON, STEP, Grid, great_circle_km


@pytest.fixture(params=[1.0, 0.25])
def dataset(request: pytest.FixtureRequest) -> xr.Dataset:
    spacing = float(request.param)
    return xr.Dataset(
        coords={
            "lat": np.arange(-90, 90 + spacing, spacing),
            "lon": np.arange(0, 360, spacing),
        }
    )


def test_constants():
    np.testing.assert_array_equal(LAT, np.arange(-90, 91, 1.0))
    np.testing.assert_array_equal(LON, np.arange(0, 360, 1.0))
    assert LEVELS_HPA == (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    assert timedelta(hours=6) == STEP
    assert GRID_1DEG.spacing_deg == 1.0
    np.testing.assert_array_equal(GRID_1DEG.lat, LAT)
    np.testing.assert_array_equal(GRID_1DEG.lon, LON)


def test_from_dataset(dataset: xr.Dataset):
    grid = Grid.from_dataset(dataset)
    assert grid.spacing_deg == float(dataset.lat[1] - dataset.lat[0])
    np.testing.assert_array_equal(grid.lat, dataset.lat)
    np.testing.assert_array_equal(grid.lon, dataset.lon)


@pytest.mark.parametrize("field", ["lat", "lon", "spacing_deg"])
def test_grid_is_frozen(field: str):
    with pytest.raises(FrozenInstanceError):
        setattr(GRID_1DEG, field, None)


@pytest.mark.parametrize(
    ("lat", "lon", "message"),
    [
        ([1, 0, -1], [0, 120, 240], "lat"),
        ([0, 1, 2], [0, 1, 3], "lon"),
        ([0, 1, 3], list(range(360)), "lat"),
        ([0, 1, 2], [0, 120, 240], "spacing"),
        ([0, 1, 2], list(range(359)), "cover"),
        ([0, 1, 2], list(range(1, 361)), "cover"),
        ([0, 1, 2], list(range(361)), "cover"),
        ([0, 0, 1], list(range(360)), "lat"),
        ([0], list(range(360)), "lat"),
        ([0, np.nan, 2], list(range(360)), "lat"),
        ([0, 1, np.inf], list(range(360)), "lat"),
    ],
)
def test_invalid_axes(lat: list[float], lon: list[float], message: str):
    with pytest.raises(ValueError, match=message):
        Grid.from_dataset(xr.Dataset(coords={"lat": lat, "lon": lon}))


def test_missing_coordinate():
    with pytest.raises(ValueError, match="lon"):
        Grid.from_dataset(xr.Dataset(coords={"lat": LAT}))


def test_multidimensional_coordinate():
    ds = xr.Dataset(coords={"lat": (("y", "x"), [[0, 1], [2, 3]]), "lon": LON})
    with pytest.raises(ValueError, match="lat"):
        Grid.from_dataset(ds)


def test_spacing_tolerance():
    lon = LON.copy()
    lon[10] += 2e-7
    assert Grid.from_dataset(xr.Dataset(coords={"lat": LAT, "lon": lon})).spacing_deg == 1.0
    lon[10] += 2e-6
    with pytest.raises(ValueError, match="lon"):
        Grid.from_dataset(xr.Dataset(coords={"lat": LAT, "lon": lon}))


def test_equatorial_degree():
    assert float(great_circle_km(0, 0, 0, 1)) == pytest.approx(111.195, abs=0.01)


latitude = st.floats(min_value=-90, max_value=90)
longitude = st.floats(min_value=-360, max_value=360)


@given(lat1=latitude, lon1=longitude, lat2=latitude, lon2=longitude)
def test_distance_properties(lat1: float, lon1: float, lat2: float, lon2: float):
    distance = float(great_circle_km(lat1, lon1, lat2, lon2))
    assert distance == pytest.approx(float(great_circle_km(lat2, lon2, lat1, lon1)), abs=1e-9)
    assert float(great_circle_km(lat1, lon1, lat1, lon1)) == 0
    assert 0 <= distance <= np.pi * 6371.0088


@pytest.mark.parametrize("lat", [-90.0, -45.0, 0.0, 45.0, 90.0])
def test_antipodes(lat: float):
    assert float(great_circle_km(lat, 0, -lat, 180)) == pytest.approx(np.pi * 6371.0088)


def test_broadcast_and_longitude_seam():
    distances = great_circle_km(0, 359, np.array([[0.0], [1.0]]), LON)
    assert distances.shape == (2, 360)
    assert distances[0, 0] == pytest.approx(111.195, abs=0.01)
    assert distances[0, 359] == 0
