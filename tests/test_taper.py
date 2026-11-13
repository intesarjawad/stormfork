import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike, NDArray

from stormfork.grid import EARTH_RADIUS_KM, GRID_1DEG, Grid, great_circle_km
from stormfork.perturb.base import Taper, anomaly, background


@pytest.fixture
def ibtracs_row() -> dict[str, str | float | None]:
    fixture = Path(__file__).parent / "fixtures" / "ibtracs_katrina_taper.csv"
    with fixture.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        next(reader)  # NOAA units row precedes the data.
        return dict(next(reader))


def test_radius_rule(ibtracs_row: dict[str, str | float | None]):
    taper = Taper.from_ibtracs(ibtracs_row)
    assert taper.centre_lat == 24.5
    assert taper.centre_lon == -76.5
    assert taper.radius_km == 400

    for radius_nm, expected_km in [(200, 555.6), (500, 900)]:
        row = {**ibtracs_row, "USA_R34_SW": radius_nm}
        assert Taper.from_ibtracs(row).radius_km == pytest.approx(expected_km)

    missing = dict(
        zip(
            ["USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"],
            [None, " ", np.nan, -9999.0],
            strict=True,
        )
    )
    with pytest.raises(ValueError, match=r"2005236N23285.*2005-08-24 12:00:00.*R34"):
        Taper.from_ibtracs({**ibtracs_row, **missing})


@pytest.mark.parametrize("quadrant", ["NE", "SE", "SW", "NW"])
def test_radius_uses_each_available_quadrant(
    ibtracs_row: dict[str, str | float | None], quadrant: str
):
    row = {**ibtracs_row, **{f"USA_R34_{name}": None for name in ["NE", "SE", "SW", "NW"]}}
    row[f"USA_R34_{quadrant}"] = 200.0
    assert Taper.from_ibtracs(row).radius_km == pytest.approx(555.6)


@pytest.mark.parametrize("field", ["centre_lat", "centre_lon", "radius_km"])
def test_taper_is_frozen(field: str):
    with pytest.raises(FrozenInstanceError):
        setattr(Taper(0, 0, 400), field, 1.0)


@pytest.mark.parametrize(
    ("lat", "lon", "radius", "message"),
    [
        (91, 0, 400, "centre_lat"),
        (-91, 0, 400, "centre_lat"),
        (np.nan, 0, 400, "centre_lat"),
        (0, np.inf, 400, "centre_lon"),
        (0, np.nan, 400, "centre_lon"),
        (0, 0, 0, "radius_km"),
        (0, 0, -400, "radius_km"),
        (0, 0, np.inf, "radius_km"),
        (0, 0, np.nan, "radius_km"),
    ],
)
def test_invalid_taper(lat: float, lon: float, radius: float, message: str):
    with pytest.raises(ValueError, match=message):
        Taper(lat, lon, radius)


@given(
    lat=st.floats(min_value=-90, max_value=90),
    lon=st.floats(min_value=-360, max_value=360),
    radius=st.floats(min_value=1, max_value=2000),
)
def test_taper_properties(lat: float, lon: float, radius: float):
    taper = Taper(lat, lon, radius)
    weights = taper.weights(GRID_1DEG)
    distances = great_circle_km(lat, lon, GRID_1DEG.lat[:, None], GRID_1DEG.lon)
    assert weights.shape == (181, 360)
    assert np.all((weights >= 0) & (weights <= 1))
    assert np.all(weights[distances <= radius / 2] == 1)
    assert np.all(weights[distances >= radius] == 0)
    centre_grid = Grid(np.array([lat]), np.array([lon]), 1.0)
    assert taper.weights(centre_grid)[0, 0] == 1


@pytest.mark.parametrize("spacing", [1.0, 0.25])
def test_cosine_profile(spacing: float):
    grid = Grid(np.arange(-1, 2, spacing), np.arange(0, 360, spacing), spacing)
    radius = float(great_circle_km(0, 0, 0, 4))
    weights = Taper(0, 0, radius).weights(grid)
    equator = int(1 / spacing)
    np.testing.assert_allclose(
        weights[equator, (np.array([0, 1, 2, 3, 4, 5]) / spacing).astype(int)],
        [1, 1, 1, 0.5, 0, 0],
        atol=1e-14,
    )


@given(
    lat=st.floats(min_value=-90, max_value=90),
    radius=st.floats(min_value=400, max_value=900),
)
def test_longitude_seam_is_continuous(lat: float, radius: float):
    grid = GRID_1DEG
    left = Taper(lat, -1e-7, radius).weights(grid)
    right = Taper(lat, 1e-7, radius).weights(grid)
    np.testing.assert_allclose(left, right, atol=2e-7, rtol=0)
    np.testing.assert_allclose(left, Taper(lat, 360 - 1e-7, radius).weights(grid), atol=1e-12)
    centred = Taper(lat, 0, radius).weights(grid)
    np.testing.assert_allclose(centred[:, 1], centred[:, -1], atol=1e-12)


def test_taper_boundaries_are_continuous():
    radius = 600.0
    offsets = np.array([0.5 - 1e-7, 0.5, 0.5 + 1e-7, 1 - 1e-7, 1, 1 + 1e-7])
    grid = Grid(np.array([0.0]), np.rad2deg(offsets * radius / EARTH_RADIUS_KM), 1.0)
    np.testing.assert_allclose(Taper(0, 0, radius).weights(grid), [[1, 1, 1, 0, 0, 0]], atol=1e-12)


@given(
    value=st.floats(min_value=-1e6, max_value=1e6),
    sigma=st.floats(min_value=0, max_value=10),
)
def test_background_of_constant(value: float, sigma: float):
    field = np.full((2, 3, 9, 18), value)
    np.testing.assert_allclose(background(field, sigma), field, atol=1e-9)
    np.testing.assert_allclose(anomaly(field, sigma), 0, atol=1e-9)


def test_background_wraps_longitude_and_reflects_latitude():
    field = np.zeros((31, 60))
    field[0, 0] = 1
    smooth = background(field, sigma_grid_cells=2)
    assert smooth[0, -1] > 0
    assert smooth[0, -1] == pytest.approx(smooth[0, 1])
    np.testing.assert_array_equal(smooth[-1], 0)
    assert smooth.sum() == pytest.approx(1)


def test_default_smoothing_width():
    field = np.zeros((81, 120))
    field[40, 60] = 1
    smooth = background(field)
    assert smooth[40, 66] / smooth[40, 60] == pytest.approx(np.exp(-0.5))


def test_background_normalises_over_defined_cells():
    field = np.full((2, 3, 17, 36), 7.0)
    field[0, 0, 8, 18] = np.nan
    field[1, 2] = np.nan
    smooth = background(field)
    np.testing.assert_array_equal(np.isnan(smooth), np.isnan(field))
    np.testing.assert_allclose(smooth[~np.isnan(field)], 7.0)
    np.testing.assert_allclose(anomaly(field)[~np.isnan(field)], 0.0, atol=1e-14)


@pytest.fixture
def field() -> NDArray[np.float64]:
    return np.random.default_rng(0).normal(size=(2, 3, 17, 36))


def test_decomposition_preserves_slices_and_input(field: NDArray[np.float64]):
    original = field.copy()
    smooth = background(field)
    residual = anomaly(field)
    assert smooth.shape == field.shape
    assert residual.shape == field.shape
    np.testing.assert_allclose(smooth + residual, field, atol=1e-14)
    np.testing.assert_array_equal(field, original)
    for frame in range(2):
        for level in range(3):
            np.testing.assert_array_equal(smooth[frame, level], background(field[frame, level]))


def test_integer_field_has_floating_background():
    field = np.zeros((9, 18), dtype=np.int64)
    field[4, 9] = 1
    smooth = background(field, sigma_grid_cells=1)
    assert smooth.dtype == np.float64
    assert 0 < smooth[4, 9] < 1
    np.testing.assert_allclose(smooth + anomaly(field, sigma_grid_cells=1), field, atol=1e-14)


def test_zero_sigma_is_identity(field: NDArray[np.float64]):
    np.testing.assert_array_equal(background(field, sigma_grid_cells=0), field)
    np.testing.assert_array_equal(anomaly(field, sigma_grid_cells=0), np.zeros_like(field))


@pytest.mark.parametrize("function", [background, anomaly])
@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf])
def test_invalid_sigma(function: Callable[[ArrayLike, float], NDArray[np.float64]], sigma: float):
    with pytest.raises(ValueError, match="sigma_grid_cells"):
        function(np.zeros((3, 6)), sigma)


@pytest.mark.parametrize("shape", [(), (3,)])
def test_background_requires_spatial_axes(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match=r"lat.*lon"):
        background(np.zeros(shape))


@pytest.mark.parametrize("column", ["USA_LAT", "USA_LON"])
@pytest.mark.parametrize("sentinel", [-9999.0, "-9999"])
def test_ibtracs_rejects_missing_position(
    ibtracs_row: dict[str, str | float | None], column: str, sentinel: str | float
):
    with pytest.raises(ValueError, match=column):
        Taper.from_ibtracs({**ibtracs_row, column: sentinel})


def test_retained_ibtracs_row_provenance():
    fixture = Path(__file__).parent / "fixtures" / "ibtracs_katrina_taper.csv"
    metadata = json.loads(fixture.with_suffix(".source.json").read_text())
    assert metadata["version"] == "v04r01"
    assert metadata["retrieved"] == "2026-11-05"
    assert metadata["source_url"].startswith("https://www.ncei.noaa.gov/data/")
    assert metadata["fixture_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    with fixture.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        units, row = list(reader)
    assert units["USA_LAT"] == "degrees_north"
    assert units["USA_LON"] == "degrees_east"
    assert units["USA_R34_NE"] == "nmile"
    assert row["SID"] == metadata["selection"]["SID"]
    assert row["ISO_TIME"] == metadata["selection"]["ISO_TIME"]
    assert row["NAME"] == "KATRINA"
    assert [row[f"USA_R34_{quadrant}"] for quadrant in ("NE", "SE", "SW", "NW")] == [
        "60",
        "60",
        "0",
        "0",
    ]
