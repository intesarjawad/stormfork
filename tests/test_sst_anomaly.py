import numpy as np
import pytest
import xarray as xr

from stormfork.grid import EARTH_RADIUS_KM, great_circle_km
from stormfork.perturb.sst_anomaly import sst_anomaly

CENTRE_LAT = 25.0
CENTRE_LON = 280.0
SIGMA_KM = 400.0
TEMPERATURE_FIELDS = ("sea_surface_temperature", "2m_temperature", "temperature")


def ocean_dataset(spacing: float = 1.0) -> xr.Dataset:
    ds = xr.Dataset(
        coords={
            "time": [0, 1],
            "level": [1000, 925, 850, 500],
            "lat": np.arange(24, 27 + spacing, spacing),
            "lon": np.arange(0, 360, spacing),
        },
        attrs={"title": "Synthetic ocean and land"},
    )
    surface_shape = (2, ds.sizes["lat"], ds.sizes["lon"])
    for name in TEMPERATURE_FIELDS:
        dims = ("time", "lat", "lon")
        shape = surface_shape
        if name == "temperature":
            dims = ("time", "level", "lat", "lon")
            shape = (2, 4, ds.sizes["lat"], ds.sizes["lon"])
        ds[name] = xr.DataArray(
            np.full(shape, 290.0, dtype=np.float32), dims=dims, attrs={"units": "K"}
        )
    mask = np.zeros(surface_shape[1:], dtype=np.float32)
    mask[0] = 1.0
    mask[-1] = 0.5
    mask[1, 0] = 0.49
    mask[1, 1] = np.nan
    ds["land_sea_mask"] = (("lat", "lon"), mask)
    ds["unrelated"] = 7
    return ds


def perturb(ds: xr.Dataset, delta_k: float = -2.0) -> xr.Dataset:
    return sst_anomaly(ds, centre=(CENTRE_LAT, CENTRE_LON), sigma_km=SIGMA_KM, delta_k=delta_k)


@pytest.mark.parametrize("spacing", [1.0, 0.25])
@pytest.mark.parametrize("delta_k", [-3.0, -2.0, 2.0, 3.0])
def test_ocean_gaussian_and_air_response(spacing: float, delta_k: float):
    ds = ocean_dataset(spacing)
    original = ds.copy(deep=True)
    result = perturb(ds, delta_k)
    distance = great_circle_km(CENTRE_LAT, CENTRE_LON, ds.lat.values[:, None], ds.lon.values)
    change = xr.DataArray(
        delta_k * np.exp(-0.5 * (distance / SIGMA_KM) ** 2),
        dims=("lat", "lon"),
        coords={"lat": ds.lat, "lon": ds.lon},
    ).where(ds.land_sea_mask < 0.5, 0)
    for name in TEMPERATURE_FIELDS:
        expected = change if name == "sea_surface_temperature" else change / 2
        if name == "temperature":
            expected = expected * ds.level.isin([1000, 925])
        expected = expected.broadcast_like(ds[name]).transpose(*ds[name].dims)
        np.testing.assert_allclose(result[name] - ds[name], expected, atol=2e-5)
        assert result[name].attrs == ds[name].attrs
    centre = result.sea_surface_temperature - ds.sea_surface_temperature
    assert float(abs(centre).max()) == pytest.approx(abs(delta_k), abs=1e-3)
    assert ds.identical(original)
    assert result.land_sea_mask.identical(ds.land_sea_mask)
    assert result.unrelated.identical(ds.unrelated)
    assert result.attrs == ds.attrs
    assert result.coords.to_dataset().identical(ds.coords.to_dataset())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_land_and_other_levels_preserve_every_bit(dtype: type[np.floating]):
    ds = ocean_dataset()
    for name in TEMPERATURE_FIELDS:
        ds[name] = ds[name].copy(data=ds[name].values.astype(dtype))
        values = ds[name].values
        values[..., 0, 0] = -0.0
        values[..., 0, 1] = np.nan
        values[..., 0, 2] = np.inf
    result = perturb(ds)
    for name in TEMPERATURE_FIELDS:
        unchanged = (~(ds.land_sea_mask < 0.5)).broadcast_like(ds[name])
        if name == "temperature":
            unchanged = unchanged | ~ds.level.isin([1000, 925])
        selected = unchanged.transpose(*ds[name].dims).values
        assert result[name].dtype == ds[name].dtype
        assert result[name].values[selected].tobytes() == ds[name].values[selected].tobytes()


def test_zero_delta_preserves_every_bit():
    ds = ocean_dataset()
    ds["temperature"].values[0, 0, 0, 0] = -0.0
    ds["sea_surface_temperature"].values[0, 0, 0] = np.nan
    result = perturb(ds, 0)
    assert result.identical(ds)
    for name in TEMPERATURE_FIELDS:
        assert result[name].values.tobytes() == ds[name].values.tobytes()


def test_off_node_centre_samples_analytic_gaussian():
    ds = ocean_dataset()
    centre = (25.5, 280.5)
    result = sst_anomaly(ds, centre=centre, sigma_km=150, delta_k=2)
    distance = great_circle_km(*centre, ds.lat.values[:, None], ds.lon.values)
    expected = 2 * np.exp(-0.5 * (distance / 150) ** 2)
    expected = np.where(ds.land_sea_mask < 0.5, expected, 0)
    change = result.sea_surface_temperature - ds.sea_surface_temperature
    np.testing.assert_allclose(change, np.broadcast_to(expected, change.shape), atol=2e-5)
    assert float(change.max()) < 2
    assert float(change.max()) == pytest.approx(float(expected.max()), abs=2e-5)


@pytest.mark.parametrize("spacing", [1.0, 0.25])
@pytest.mark.parametrize("delta_k", [0.0, 2.0])
def test_sigma_must_resolve_at_least_one_grid_spacing(spacing: float, delta_k: float):
    ds = ocean_dataset(spacing)
    spacing_km = EARTH_RADIUS_KM * np.deg2rad(spacing)
    with pytest.raises(ValueError, match=r"sigma_km.*grid spacing"):
        sst_anomaly(
            ds, centre=(CENTRE_LAT, CENTRE_LON), sigma_km=0.99 * spacing_km, delta_k=delta_k
        )
    sst_anomaly(ds, centre=(CENTRE_LAT, CENTRE_LON), sigma_km=spacing_km, delta_k=delta_k)


def test_dimension_order_and_custom_names():
    ds = ocean_dataset()
    names = {name: f"custom_{name}" for name in (*TEMPERATURE_FIELDS, "land_sea_mask")}
    renamed = ds.rename(names).transpose("lon", "time", "lat", "level")
    result = sst_anomaly(
        renamed,
        centre=(CENTRE_LAT, CENTRE_LON),
        sigma_km=SIGMA_KM,
        delta_k=-2,
        names=names,
    )
    expected = perturb(ds).rename(names)
    for name in names.values():
        assert result[name].dims == renamed[name].dims
        assert result[name].transpose(*expected[name].dims).identical(expected[name])


def test_longitude_wrap_and_frames():
    ds = ocean_dataset()
    ds["land_sea_mask"][:] = 0
    result = sst_anomaly(ds, centre=(25, 0), sigma_km=400, delta_k=2)
    negative_lon = sst_anomaly(ds, centre=(25, -360), sigma_km=400, delta_k=2)
    np.testing.assert_array_equal(result.temperature.values, negative_lon.temperature.values)
    np.testing.assert_array_equal(
        result.sea_surface_temperature.sel(lon=1), result.sea_surface_temperature.sel(lon=359)
    )
    for time in ds.time.values:
        assert perturb(ds.sel(time=time)).identical(perturb(ds).sel(time=time))


@pytest.mark.parametrize("delta_k", [-3.01, 3.01, np.nan, np.inf, -np.inf])
def test_invalid_delta(delta_k: float):
    with pytest.raises(ValueError, match="delta_k"):
        perturb(ocean_dataset(), delta_k)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("sigma_km", 0),
        ("sigma_km", -1),
        ("sigma_km", np.inf),
        ("sigma_km", np.nan),
        ("centre_lat", 91),
        ("centre_lat", -91),
        ("centre_lat", np.nan),
        ("centre_lon", np.inf),
        ("centre_lon", np.nan),
    ],
)
def test_invalid_geometry(parameter: str, value: float):
    arguments = {"centre_lat": 25.0, "centre_lon": 280.0, "sigma_km": 400.0}
    arguments[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        sst_anomaly(
            ocean_dataset(),
            centre=(arguments["centre_lat"], arguments["centre_lon"]),
            sigma_km=arguments["sigma_km"],
            delta_k=2,
        )


@pytest.mark.parametrize("delta_k", [0.0, 2.0])
def test_invalid_grid(delta_k: float):
    with pytest.raises(ValueError, match="lon"):
        perturb(ocean_dataset().isel(lon=slice(1, None)), delta_k)
