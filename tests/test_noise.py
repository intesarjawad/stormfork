import numpy as np
import pytest
import xarray as xr

from stormfork.perturb.noise import noise


def weather_dataset(spacing: float = 1.0) -> xr.Dataset:
    lat = np.arange(-20, 21, spacing)
    lon = np.arange(0, 360, spacing)
    pattern = np.sin(np.deg2rad(lon))[None, :] + lat[:, None] / 20
    temperature = 290 + np.array([[1, 10], [2, 20]])[:, :, None, None] * pattern
    return xr.Dataset(
        {
            "temperature": (("time", "level", "lat", "lon"), temperature, {"units": "K"}),
            "sea_surface_temperature": (("lat", "lon"), 290 + pattern),
            "land_sea_mask": (("lat", "lon"), np.zeros_like(pattern)),
            "unrelated": 7,
        },
        coords={"time": [0, 1], "level": [1000, 500], "lat": lat, "lon": lon},
        attrs={"title": "Synthetic weather"},
    )


@pytest.mark.parametrize("spacing", [1.0, 0.25])
@pytest.mark.parametrize("amplitude", [0.01, 0.05])
def test_empirical_standard_deviation(spacing: float, amplitude: float):
    ds = weather_dataset(spacing)
    result = noise(ds, amplitude=amplitude, seed=42)
    for name in ("temperature", "sea_surface_temperature"):
        change = result[name] - ds[name]
        target = amplitude * ds[name].std(dim=("lat", "lon"))
        np.testing.assert_allclose(change.std(dim=("lat", "lon")), target, rtol=0.1)
        assert np.all(abs(change.mean(dim=("lat", "lon"))) < target * 0.1)
    normalized = (result.temperature - ds.temperature) / ds.temperature.std(dim=("lat", "lon"))
    assert not np.array_equal(normalized[0, 0], normalized[0, 1])
    assert not np.array_equal(normalized[0, 0], normalized[1, 0])


def test_seed_determinism_and_input_preservation():
    ds = weather_dataset()
    original = ds.copy(deep=True)
    result = noise(ds, amplitude=0.05, seed=3)
    assert result.identical(noise(ds, amplitude=0.05, seed=3))
    assert not result.temperature.identical(noise(ds, amplitude=0.05, seed=4).temperature)
    assert ds.identical(original)
    assert result.land_sea_mask.identical(ds.land_sea_mask)
    assert result.unrelated.identical(ds.unrelated)
    assert result.attrs == ds.attrs
    assert result.coords.to_dataset().identical(ds.coords.to_dataset())
    assert result.temperature.attrs == ds.temperature.attrs


def test_missing_and_constant_slices():
    ds = weather_dataset()
    ds.temperature.values[0, 0] = np.nan
    ds.temperature.values[0, 1] = -0.0
    ds.sea_surface_temperature.values[0] = np.nan
    result = noise(ds, amplitude=0.05, seed=7)
    assert result.temperature.values[0].tobytes() == ds.temperature.values[0].tobytes()
    np.testing.assert_array_equal(
        np.isnan(result.sea_surface_temperature), np.isnan(ds.sea_surface_temperature)
    )
    target = 0.05 * float(ds.sea_surface_temperature.std())
    assert float(
        (result.sea_surface_temperature - ds.sea_surface_temperature).std()
    ) == pytest.approx(target, rel=0.1)


def test_zero_amplitude_preserves_every_bit():
    ds = weather_dataset()
    ds["temperature"] = ds.temperature.astype(np.float32)
    ds.temperature.values[0, 0, 0, 0] = -0.0
    ds.temperature.values[0, 0, 0, 1] = np.nan
    result = noise(ds, amplitude=0, seed=0)
    assert result.identical(ds)
    assert result.temperature.values.tobytes() == ds.temperature.values.tobytes()


def test_dimension_order_and_custom_names():
    ds = weather_dataset()
    renamed = ds.rename({"temperature": "t"}).transpose("lon", "time", "lat", "level")
    result = noise(renamed, amplitude=0.05, seed=42, names={"temperature": "t"})
    expected = noise(ds, amplitude=0.05, seed=42).rename({"temperature": "t"})
    assert result.t.dims == renamed.t.dims
    assert result.transpose(*expected.dims).identical(expected)


def test_seed_is_independent_of_time_level_dimension_order():
    ds = weather_dataset()
    transposed = ds.transpose("level", "lon", "time", "lat")
    result = noise(transposed, amplitude=0.05, seed=42)
    expected = noise(ds, amplitude=0.05, seed=42)
    for name in ds.data_vars:
        assert result[name].dims == transposed[name].dims
        assert result[name].transpose(*ds[name].dims).identical(expected[name])


@pytest.mark.parametrize("amplitude", [-0.01, 0.051, np.inf, -np.inf, np.nan])
def test_invalid_amplitude(amplitude: float):
    with pytest.raises(ValueError, match="amplitude"):
        noise(weather_dataset(), amplitude=amplitude, seed=0)


@pytest.mark.parametrize("amplitude", [0, 0.05])
def test_invalid_grid(amplitude: float):
    with pytest.raises(ValueError, match="lon"):
        noise(weather_dataset().isel(lon=slice(1, None)), amplitude=amplitude, seed=0)
