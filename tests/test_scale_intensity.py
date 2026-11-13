import numpy as np
import pytest

from stormfork.grid import Grid
from stormfork.perturb.base import background
from stormfork.perturb.scale_intensity import scale_intensity
from tests.test_relocate import CENTRE_LAT, CENTRE_LON, SURFACE, TAPER, UPPER_AIR, gaussian_low


@pytest.mark.parametrize("spacing", [1.0, 0.25])
@pytest.mark.parametrize("factor", [0.5, 0.7, 1.5])
def test_scale_gaussian_low(spacing: float, factor: float):
    ds = gaussian_low(spacing)
    original = ds.copy(deep=True)
    result = scale_intensity(ds, TAPER, factor=factor)
    centre_row = int(np.flatnonzero(ds.lat.values == CENTRE_LAT)[0])
    centre_column = int(np.flatnonzero(ds.lon.values == CENTRE_LON)[0])
    for name, (baseline, _) in (UPPER_AIR | SURFACE).items():
        before = ds[name].isel(lat=centre_row, lon=centre_column)
        after = result[name].isel(lat=centre_row, lon=centre_column)
        np.testing.assert_allclose((baseline - after) / (baseline - before), factor, rtol=0.05)
        assert result[name].dims == ds[name].dims
        assert result[name].attrs == ds[name].attrs
    for name in set(ds.data_vars) - (UPPER_AIR | SURFACE).keys():
        assert result[name].identical(ds[name])
    assert ds.identical(original)
    assert result.attrs == ds.attrs
    assert result.coords.to_dataset().identical(ds.coords.to_dataset())


def test_factor_one_is_bitwise_identity():
    ds = gaussian_low()
    for name in ds.data_vars:
        ds[name] = ds[name].copy(data=ds[name].values.astype(np.float32))
    ds["temperature"][0, 0, 0, 0] = np.float32(-0.0)
    ds["temperature"][1, 1, 0, 0] = np.nan
    result = scale_intensity(ds, TAPER, factor=1.0)
    assert result.identical(ds)
    for name in ds.data_vars:
        assert result[name].values.dtype == ds[name].values.dtype
        assert result[name].values.tobytes() == ds[name].values.tobytes()


def test_scaling_preserves_nan_mask_inside_and_outside_taper():
    ds = gaussian_low()
    row = int(np.flatnonzero(ds.lat.values == CENTRE_LAT)[0])
    columns = np.flatnonzero(np.isin(ds.lon.values, [CENTRE_LON, 0]))
    for name in UPPER_AIR | SURFACE:
        ds[name].values[..., row, columns] = np.nan
    original = ds.copy(deep=True)
    result = scale_intensity(ds, TAPER, factor=1.5)
    for name in UPPER_AIR | SURFACE:
        np.testing.assert_array_equal(np.isnan(result[name]), np.isnan(ds[name]))
    assert ds.identical(original)


def test_scaling_follows_taper():
    ds = gaussian_low()
    factor = 1.5
    result = scale_intensity(ds, TAPER, factor=factor)
    weights = TAPER.weights(Grid.from_dataset(ds))
    assert np.any((weights > 0) & (weights < 1))
    for name in UPPER_AIR | SURFACE:
        before = ds[name].values
        expected_change = (before - background(before)) * (factor - 1) * weights
        np.testing.assert_allclose(result[name].values - before, expected_change, atol=1e-10)


def test_dimension_order_and_custom_names():
    ds = gaussian_low().rename({"temperature": "t"}).transpose("lon", "time", "lat", "level")
    result = scale_intensity(ds, TAPER, factor=0.7, names={"temperature": "t"})
    expected = scale_intensity(gaussian_low(), TAPER, factor=0.7).temperature
    np.testing.assert_allclose(result.t.transpose(*expected.dims).values, expected.values)
    assert result.t.dims == ds.t.dims


def test_each_frame_and_level_scales_independently():
    ds = gaussian_low()
    result = scale_intensity(ds, TAPER, factor=0.7)
    for time in range(2):
        for level in range(2):
            expected = scale_intensity(ds.isel(time=time, level=level), TAPER, factor=0.7)
            assert result.isel(time=time, level=level).identical(expected)


@pytest.mark.parametrize("factor", [0.49, 1.51, np.nan, np.inf, -np.inf])
def test_invalid_factor(factor: float):
    with pytest.raises(ValueError, match=r"factor.*finite.*0\.5.*1\.5"):
        scale_intensity(gaussian_low(), TAPER, factor=factor)


@pytest.mark.parametrize("factor", [0.7, 1.0])
def test_invalid_grid(factor: float):
    with pytest.raises(ValueError, match="lon"):
        scale_intensity(gaussian_low().isel(lon=slice(1, None)), TAPER, factor=factor)
