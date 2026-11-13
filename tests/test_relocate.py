import numpy as np
import pytest
import xarray as xr

from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.perturb.base import Taper, background
from stormfork.perturb.relocate import relocate

CENTRE_LAT = 25.0
CENTRE_LON = 280.0
TAPER = Taper(CENTRE_LAT, CENTRE_LON, 900)
UPPER_AIR = {
    "temperature": (280.0, 3.0),
    "geopotential": (50000.0, 300.0),
    "u_component_of_wind": (10.0, 2.0),
    "v_component_of_wind": (-8.0, 2.0),
    "vertical_velocity": (0.1, 0.02),
    "specific_humidity": (0.01, 0.002),
}
SURFACE = {
    "2m_temperature": (298.0, 3.0),
    "mean_sea_level_pressure": (101000.0, 6000.0),
    "10m_u_component_of_wind": (12.0, 3.0),
    "10m_v_component_of_wind": (-6.0, 3.0),
}


def gaussian_low(
    spacing: float = 1.0,
    centre_lon: float = CENTRE_LON,
    centre_lat: float = CENTRE_LAT,
) -> xr.Dataset:
    ds = xr.Dataset(
        coords={
            "time": np.array(["2012-10-25T18", "2012-10-26T00"], dtype="datetime64[ns]"),
            "level": [500, 850],
            "lat": np.arange(centre_lat - 15, centre_lat + 15 + spacing, spacing),
            "lon": np.arange(0, 360, spacing),
        },
        attrs={"title": "Synthetic Gaussian low"},
    )
    grid = Grid.from_dataset(ds)
    distance = great_circle_km(centre_lat, centre_lon, grid.lat[:, None], grid.lon)
    profile = np.exp(-0.5 * (distance / 130) ** 2)
    frames = np.array([1.0, 1.2])[:, None, None] * profile
    levels = np.array([1.0, 0.8])[None, :, None, None] * frames[:, None]
    for name, (baseline, depth) in UPPER_AIR.items():
        ds[name] = xr.DataArray(
            baseline - depth * levels,
            dims=("time", "level", "lat", "lon"),
            attrs={"long_name": name},
        )
    for name, (baseline, depth) in SURFACE.items():
        ds[name] = xr.DataArray(
            baseline - depth * frames,
            dims=("time", "lat", "lon"),
            attrs={"long_name": name},
        )
    ds["sea_surface_temperature"] = (("time", "lat", "lon"), 299 - frames)
    ds["sea_surface_temperature"][0, 0, 0] = np.nan
    ds["land_sea_mask"] = (("lat", "lon"), (distance < 100).astype(np.float32))
    ds["geopotential_at_surface"] = (("lat", "lon"), distance.astype(np.float32))
    ds["unrelated"] = 7
    return ds


@pytest.mark.parametrize("spacing", [1.0, 0.25])
@pytest.mark.parametrize("centre_lat", [15.0, 30.0, 60.0])
@pytest.mark.parametrize(
    ("east_km", "north_km"), [(300, 0), (-300, 0), (0, 300), (0, -300), (180, 240)]
)
def test_relocate_gaussian_low(spacing: float, centre_lat: float, east_km: float, north_km: float):
    ds = gaussian_low(spacing, centre_lat=centre_lat)
    original = ds.copy(deep=True)
    moved = relocate(ds, Taper(centre_lat, CENTRE_LON, 900), east_km=east_km, north_km=north_km)
    cell_diagonal_km = float(
        great_circle_km(centre_lat, CENTRE_LON, centre_lat + spacing, CENTRE_LON + spacing)
    )
    for name in UPPER_AIR | SURFACE:
        before = ds[name]
        after = moved[name]
        assert not np.array_equal(before.values, after.values), name
        before_slices = before.values.reshape(-1, before.sizes["lat"], before.sizes["lon"])
        after_slices = after.values.reshape(-1, after.sizes["lat"], after.sizes["lon"])
        for source, field in zip(before_slices, after_slices, strict=True):
            source_row, source_column = np.unravel_index(np.argmin(source), source.shape)
            row, column = np.unravel_index(np.argmin(field), field.shape)
            distance_km = float(
                great_circle_km(
                    ds.lat.values[source_row],
                    ds.lon.values[source_column],
                    ds.lat.values[row],
                    ds.lon.values[column],
                )
            )
            assert distance_km == pytest.approx(
                np.hypot(east_km, north_km), abs=cell_diagonal_km / 2
            ), name
            assert np.sign(row - source_row) == np.sign(north_km), name
            assert np.sign(column - source_column) == np.sign(east_km), name
        assert after.dims == before.dims
        assert after.attrs == before.attrs
    for name in set(ds.data_vars) - (UPPER_AIR | SURFACE).keys():
        assert moved[name].identical(ds[name])
    assert ds.identical(original)
    assert moved.attrs == ds.attrs
    assert moved.coords.to_dataset().identical(ds.coords.to_dataset())


def test_zero_displacement_is_bitwise_identity():
    ds = gaussian_low()
    for name in ds.data_vars:
        ds[name] = ds[name].copy(data=ds[name].values.astype(np.float32))
    ds["temperature"][0, 0, 0, 0] = np.float32(-0.0)
    ds["temperature"][1, 1, 0, 0] = np.nan
    result = relocate(ds, TAPER, east_km=0, north_km=0)
    assert result.identical(ds)
    for name in ds.data_vars:
        assert result[name].values.dtype == ds[name].values.dtype
        assert result[name].values.tobytes() == ds[name].values.tobytes()


@pytest.mark.parametrize("east_km", [-300.0, -55.0, 55.0, 300.0])
def test_fractional_shift_wraps_longitude(east_km: float):
    ds = gaussian_low()
    moved = relocate(ds, TAPER, east_km=east_km)
    seam_ds = ds.roll(lon=80, roll_coords=False)
    seam_moved = relocate(seam_ds, Taper(CENTRE_LAT, 0, 900), east_km=east_km)
    for name in UPPER_AIR | SURFACE:
        np.testing.assert_allclose(
            seam_moved[name].values, moved[name].roll(lon=80, roll_coords=False).values
        )
        np.testing.assert_allclose(
            seam_moved[name].mean(("lat", "lon")),
            seam_ds[name].mean(("lat", "lon")),
            rtol=1e-12,
        )


def test_each_frame_and_level_moves_independently():
    ds = gaussian_low()
    moved = relocate(ds, TAPER, east_km=230, north_km=-110)
    for time in range(2):
        for level in range(2):
            sliced = ds.isel(time=time, level=level)
            expected = relocate(sliced, TAPER, east_km=230, north_km=-110)
            assert moved.isel(time=time, level=level).identical(expected)


def test_dimension_order_and_custom_names():
    ds = gaussian_low().rename({"temperature": "t"}).transpose("lon", "time", "lat", "level")
    moved = relocate(ds, TAPER, east_km=300, names={"temperature": "t"})
    expected = relocate(gaussian_low(), TAPER, east_km=300).temperature
    np.testing.assert_allclose(moved.t.transpose(*expected.dims).values, expected.values)
    assert moved.t.dims == ds.t.dims


@pytest.mark.parametrize(
    ("east_km", "north_km"),
    [(601, 0), (-601, 0), (0, 601), (500, 500), (np.nan, 0), (0, np.inf)],
)
def test_invalid_displacement(east_km: float, north_km: float):
    with pytest.raises(ValueError, match=r"displacement.*600 km"):
        relocate(gaussian_low(), TAPER, east_km=east_km, north_km=north_km)


@pytest.mark.parametrize(("east_km", "north_km"), [(600, 0), (0, -600), (360, 480)])
def test_displacement_limit_is_inclusive(east_km: float, north_km: float):
    relocate(gaussian_low(), TAPER, east_km=east_km, north_km=north_km)


def test_invalid_grid():
    with pytest.raises(ValueError, match="lon"):
        relocate(gaussian_low().isel(lon=slice(1, None)), TAPER, east_km=300)


@pytest.mark.parametrize("cells", [1.0, 1.5, -1.5])
def test_tapered_anomaly_translation(cells: float):
    ds = gaussian_low(centre_lon=0)
    taper = Taper(CENTRE_LAT, 0, 900)
    east_km = cells * EARTH_RADIUS_KM * np.deg2rad(1) * np.cos(np.deg2rad(CENTRE_LAT))
    result = relocate(ds, taper, east_km=float(east_km))
    original = ds.mean_sea_level_pressure.values
    core = (original - background(original)) * taper.weights(Grid.from_dataset(ds))
    lower = int(np.floor(cells))
    fraction = cells - lower
    translated = (1 - fraction) * np.roll(core, lower, axis=-1) + fraction * np.roll(
        core, lower + 1, axis=-1
    )
    np.testing.assert_allclose(
        result.mean_sea_level_pressure.values, original - core + translated, atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("spacing", [1.0, 0.25])
def test_northward_translation_has_cosine_area_ratio(spacing: float):
    ds = gaussian_low(spacing)
    destination_lat = 30.0
    north_km = float(great_circle_km(CENTRE_LAT, CENTRE_LON, destination_lat, CENTRE_LON))
    result = relocate(ds, TAPER, north_km=north_km)
    before = ds.mean_sea_level_pressure.values
    smooth = background(before, sigma_grid_cells=6 / spacing)
    core = (before - smooth) * TAPER.weights(Grid.from_dataset(ds))
    transported = result.mean_sea_level_pressure.values - (before - core)
    cells = int((destination_lat - CENTRE_LAT) / spacing)
    expected = np.zeros_like(core)
    expected[..., cells:, :] = core[..., :-cells, :]
    np.testing.assert_allclose(transported, expected, atol=1e-9, rtol=0)
    south = np.deg2rad(ds.lat.values - spacing / 2)
    north = np.deg2rad(ds.lat.values + spacing / 2)
    areas_km2 = EARTH_RADIUS_KM**2 * np.deg2rad(spacing) * (np.sin(north) - np.sin(south))
    source_integral = (core * areas_km2[:, None]).sum(axis=(-2, -1))
    destination_integral = (transported * areas_km2[:, None]).sum(axis=(-2, -1))
    source_to_destination_ratio = np.cos(np.deg2rad(CENTRE_LAT)) / np.cos(
        np.deg2rad(destination_lat)
    )
    np.testing.assert_allclose(
        source_integral / destination_integral, source_to_destination_ratio, rtol=0.01
    )


@pytest.mark.parametrize(
    ("centre_lat", "east_km", "north_km"), [(90, 1, 0), (-90, -1, 0), (89, 0, 300), (-89, 0, -300)]
)
def test_invalid_polar_displacement(centre_lat: float, east_km: float, north_km: float):
    with pytest.raises(ValueError, match=r"pole|latitude"):
        relocate(gaussian_low(), Taper(centre_lat, 0, 900), east_km=east_km, north_km=north_km)


def test_relocation_preserves_nan_mask_inside_and_outside_taper():
    ds = gaussian_low()
    row = int(np.flatnonzero(ds.lat.values == CENTRE_LAT)[0])
    columns = np.flatnonzero(np.isin(ds.lon.values, [CENTRE_LON, 0]))
    for name in UPPER_AIR | SURFACE:
        ds[name].values[..., row, columns] = np.nan
    original = ds.copy(deep=True)
    moved = relocate(ds, TAPER, east_km=300, north_km=100)
    for name in UPPER_AIR | SURFACE:
        np.testing.assert_array_equal(np.isnan(moved[name]), np.isnan(ds[name]))
    assert ds.identical(original)


@pytest.mark.parametrize("hemisphere", [-1, 1])
@pytest.mark.parametrize(("centre_lat", "north_km"), [(82, 0), (80, 300)])
def test_taper_footprint_must_stay_equatorward_of_85_degrees(
    hemisphere: int, centre_lat: float, north_km: float
):
    ds = xr.Dataset(coords={"lat": np.arange(-90, 91), "lon": np.arange(360)})
    with pytest.raises(ValueError, match=r"85.*latitude|latitude.*85"):
        relocate(
            ds, Taper(hemisphere * centre_lat, 0, 400), east_km=10, north_km=hemisphere * north_km
        )


@pytest.mark.parametrize(
    ("south", "north", "north_km"),
    [
        (20, 40, 0),
        (10, 30, 0),
        (23, 27, 0),
        (15, 35, -300),
        (15, 35, 300),
    ],
)
def test_taper_footprints_must_fit_regional_latitudes(south: int, north: int, north_km: float):
    ds = gaussian_low().sel(lat=slice(south, north))
    with pytest.raises(ValueError, match=r"latitude.*range|range.*latitude"):
        relocate(ds, TAPER, north_km=north_km)
