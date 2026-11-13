from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pytest
import xarray as xr
from pydantic import BaseModel
from weathernext.utils.data_utils import extract_inputs_targets_forcings
from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name

from stormfork.grid import LAT, LEVELS_HPA, LON
from stormfork.ic.era5 import ATMOSPHERE, CYCLONE_TARGETS, FORCINGS, SURFACE, TARGETS, build_input
from stormfork.ic.layout import STATIC_NAMES, SampleStatics, coarsen, load_sample_statics

INIT = datetime(2024, 10, 7, tzinfo=UTC)
SAMPLE_PATH = Path("data/model/hres_2024-10-07_01.nc")
ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


class NetcdfWritable(Protocol):
    def to_netcdf(self, path: Path, *, engine: Literal["h5netcdf"]) -> None: ...


class VariableHeader(BaseModel):
    dims: list[str]
    dtype: str


class SampleHeader(BaseModel):
    sizes: dict[str, int]
    variables: dict[str, VariableHeader]
    coordinates: dict[str, list[float] | list[str] | list[list[str]]]


@pytest.fixture(scope="module")
def statics() -> SampleStatics:
    coords = {"lat": LAT.astype(np.float32), "lon": LON.astype(np.float32)}
    return SampleStatics(
        xr.DataArray(np.full((181, 360), 12, dtype=np.float32), dims=("lat", "lon"), coords=coords),
        xr.DataArray(
            np.full((181, 360), 0.5, dtype=np.float32), dims=("lat", "lon"), coords=coords
        ),
    )


@pytest.fixture(scope="module")
def store() -> xr.Dataset:
    atmosphere_shape = (2, 13, 721, 1440)
    surface_shape = (2, 721, 1440)
    lat = np.arange(90, -90.25, -0.25)
    levels = np.array(LEVELS_HPA[::-1])
    surface_values = np.arange(2)[:, None, None] * 100 + lat[None, :, None] / 100
    atmosphere_values = surface_values[:, None, :, :] + levels[None, :, None, None] / 1000
    fields = {
        name: xr.DataArray(
            np.broadcast_to(index + atmosphere_values, atmosphere_shape),
            dims=("time", "level", "latitude", "longitude"),
        )
        for index, name in enumerate(ATMOSPHERE)
    }
    fields.update(
        {
            name: xr.DataArray(
                np.broadcast_to(index + surface_values, surface_shape),
                dims=("time", "latitude", "longitude"),
            )
            for index, name in enumerate(SURFACE)
        }
    )
    return xr.Dataset(
        fields,
        coords={
            "time": np.array(["2024-10-06T18", "2024-10-07T00"], dtype="datetime64[ns]"),
            "level": levels,
            "latitude": lat,
            "longitude": np.arange(0, 360, 0.25),
        },
    )


@pytest.fixture(scope="module")
def built(store: xr.Dataset, statics: SampleStatics) -> xr.Dataset:
    return build_input(INIT, store, statics=statics)


def test_layout_matches_official_header(built: xr.Dataset):
    header = SampleHeader.model_validate_json(Path("tests/fixtures/hres_header.json").read_text())
    assert dict(built.sizes) == header.sizes | {"time": 42}
    assert set(built.variables) == set(header.variables)
    for name, expected in header.variables.items():
        assert built[name].dims == tuple(expected.dims), name
        assert str(built[name].dtype) == expected.dtype, name
    for name in ("lat", "lon", "level"):
        np.testing.assert_array_equal(built[name].values, header.coordinates[name])
    for name in ("time", "datetime"):
        np.testing.assert_array_equal(
            built[name].isel(time=slice(0, 3)).values.astype(str), header.coordinates[name]
        )
    np.testing.assert_array_equal(built.time.values, np.arange(42) * np.timedelta64(6, "h"))
    np.testing.assert_array_equal(
        built.datetime.values[0], np.datetime64("2024-10-06T18", "ns") + built.time.values
    )
    np.testing.assert_array_equal(
        built.seconds_since_epoch.values,
        built.datetime.values.astype("datetime64[s]").astype(np.int64),
    )


def test_inputs_statics_and_nan_targets(built: xr.Dataset, statics: SampleStatics):
    task = get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini").task
    for names in (ATMOSPHERE, SURFACE):
        for index, name in enumerate(names):
            expected = index + np.arange(2)[:, None, None] * 100 + LAT[None, :, None] / 100
            if name in ATMOSPHERE:
                expected = (
                    expected[:, None, :, :] + np.array(LEVELS_HPA)[None, :, None, None] / 1000
                )
            np.testing.assert_array_equal(
                built[name].isel(time=slice(0, 2)).values[0],
                np.broadcast_to(
                    expected.astype(np.float32), built[name].isel(batch=0, time=slice(0, 2)).shape
                ),
            )
    for name in task.target_variables:
        assert np.isnan(built[name].isel(time=slice(2, None)).values).all(), name
    for name, field in zip(
        STATIC_NAMES, (statics.geopotential_at_surface, statics.land_sea_mask), strict=True
    ):
        np.testing.assert_array_equal(built[name].values, field.values)
        assert not np.shares_memory(built[name].values, field.values)
    inputs, targets, forcings = extract_inputs_targets_forcings(
        built,
        input_variables=task.input_variables,
        target_variables=task.target_variables,
        forcing_variables=task.forcing_variables,
        pressure_levels=task.pressure_levels,
        input_duration=task.input_duration,
        target_lead_times=slice("6h", "240h"),
    )
    assert inputs.sizes["time"] == 2
    assert targets.sizes["time"] == forcings.sizes["time"] == 40
    np.testing.assert_array_equal(targets.time.values, np.arange(1, 41) * np.timedelta64(6, "h"))
    for name in task.forcing_variables:
        assert np.isfinite(inputs[name].values).all()
        np.testing.assert_array_equal(
            forcings[name].values, built[name].isel(time=slice(2, None)).values
        )


def test_explicit_statics_and_pure_inputs(
    store: xr.Dataset, statics: SampleStatics, monkeypatch: pytest.MonkeyPatch
):
    def reject_file_loading(*_args: object, **_kwargs: object) -> xr.Dataset:
        msg = "The builder must not open files"
        raise AssertionError(msg)

    monkeypatch.setattr(xr, "open_dataset", reject_file_loading)
    monkeypatch.setattr(xr, "load_dataset", reject_file_loading)
    monkeypatch.setattr(xr, "open_zarr", reject_file_loading)
    replacement = SampleStatics(
        statics.geopotential_at_surface + np.float32(7), statics.land_sea_mask
    )
    local_init = INIT.astimezone(timezone(timedelta(hours=-5)))
    actual = build_input(local_init, store, 1, statics=replacement)
    assert actual.sizes["time"] == 3
    np.testing.assert_array_equal(actual.geopotential_at_surface.values, np.float32(19))
    np.testing.assert_array_equal(statics.geopotential_at_surface.values, np.float32(12))
    assert "latitude" in store.coords
    assert "lat" not in store.coords
    assert "datetime" not in store.coords
    assert "year_progress_sin" not in store
    assert store.temperature.isel(time=0, level=0, latitude=0, longitude=0).item() == pytest.approx(
        1.9
    )


@pytest.fixture
def writable_statics(statics: SampleStatics) -> NetcdfWritable:
    return xr.Dataset(
        {
            "geopotential_at_surface": statics.geopotential_at_surface,
            "land_sea_mask": statics.land_sea_mask,
        }
    )


def test_loader_closes_sample(
    tmp_path: Path, statics: SampleStatics, writable_statics: NetcdfWritable
):
    path = tmp_path / "sample.nc"
    writable_statics.to_netcdf(path, engine="h5netcdf")
    loaded = load_sample_statics(path)
    path.unlink()
    np.testing.assert_array_equal(
        loaded.geopotential_at_surface.values, statics.geopotential_at_surface.values
    )
    np.testing.assert_array_equal(loaded.land_sea_mask.values, statics.land_sea_mask.values)


def test_builder_contract_matches_pinned_model():
    task = get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini").task
    assert set(task.input_variables) == set(ATMOSPHERE + SURFACE + STATIC_NAMES + FORCINGS)
    assert set(task.target_variables) == set(TARGETS)
    assert len(CYCLONE_TARGETS) == 17
    assert task.forcing_variables == FORCINGS


def test_statics_reject_wrong_layout(statics: SampleStatics):
    with pytest.raises(ValueError, match="float32"):
        SampleStatics(
            statics.geopotential_at_surface.copy(
                data=np.asarray(statics.geopotential_at_surface.values, dtype=np.float64)
            ),
            statics.land_sea_mask,
        )
    with pytest.raises(ValueError, match="ascending"):
        SampleStatics(
            statics.geopotential_at_surface.isel(lat=slice(None, None, -1)), statics.land_sea_mask
        )


@pytest.mark.parametrize("num_steps", [0, -1])
def test_reject_nonpositive_steps(store: xr.Dataset, statics: SampleStatics, num_steps: int):
    with pytest.raises(ValueError, match="positive"):
        build_input(INIT, store, num_steps, statics=statics)


def test_reject_off_cycle(store: xr.Dataset, statics: SampleStatics):
    with pytest.raises(ValueError, match="six-hour"):
        build_input(INIT + timedelta(minutes=1), store, statics=statics)


def test_missing_time_is_not_interpolated(store: xr.Dataset, statics: SampleStatics):
    with pytest.raises(KeyError):
        build_input(INIT + timedelta(hours=6), store, statics=statics)


def test_coarsen_subsample_and_weighted_cell_mean():
    lat = np.arange(90, -90.25, -0.25)
    lon = np.arange(0, 360, 0.25)
    values = np.broadcast_to(lat[:, None] ** 2, (721, 1440)).copy()
    values[:, -1] += 16
    source = xr.Dataset({"field": (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon})
    sampled = coarsen(source)
    np.testing.assert_array_equal(sampled.field.values, LAT[:, None] ** 2 + np.zeros((181, 360)))
    averaged = coarsen(source, "cell_mean")
    offsets = np.arange(-0.5, 0.75, 0.25)
    trapezoid = np.array([0.5, 1, 1, 1, 0.5])
    weights = np.cos(np.deg2rad(45 + offsets)) * trapezoid
    expected = np.average((45 + offsets) ** 2, weights=weights)
    assert averaged.field.sel(lat=45, lon=20).item() == pytest.approx(expected)
    assert averaged.field.sel(lat=45, lon=0).item() == pytest.approx(expected + 4)
    polar_weights = np.cos(np.deg2rad(90 + offsets[:3])) * trapezoid[:3]
    assert averaged.field.sel(lat=90, lon=20).item() == pytest.approx(
        np.average((90 + offsets[:3]) ** 2, weights=polar_weights)
    )
    assert np.isfinite(averaged.field.values).all()
    np.testing.assert_array_equal(source.lat.values, lat)
    with pytest.raises(ValueError, match="quarter-degree"):
        coarsen(sampled)


@pytest.mark.network
def test_era5_rms_against_hres():
    sample = xr.load_dataset(SAMPLE_PATH, engine="h5netcdf")
    statics = load_sample_statics(SAMPLE_PATH)
    with xr.open_zarr(
        ARCO_STORE, consolidated=True, storage_options={"token": "anon"}, chunks=None
    ) as remote:
        actual = build_input(INIT, remote, statics=statics)
    np.testing.assert_array_equal(
        actual.datetime.isel(time=slice(0, 2)).values,
        sample.datetime.isel(time=slice(0, 2)).values,
    )
    lines = [
        "## ERA5 initial-condition comparison",
        "",
        "Initialization: 2024-10-07 00:00 UTC. Coarsening: subsample.",
        "",
        "RMS differences use both input frames, finite paired cells and all model levels.",
        "The offline header fixture was extracted from the verified local HRES sample above.",
        "",
        (
            "Reproduce with `UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run pytest -m network "
            "tests/test_era5_layout.py`."
        ),
        "",
        "| Variable | RMS difference (native units) |",
        "|---|---:|",
    ]
    for name in ATMOSPHERE + SURFACE:
        difference = (
            actual[name].isel(time=slice(0, 2)).values.astype(np.float64)
            - sample[name].isel(time=slice(0, 2)).values
        )
        finite = difference[np.isfinite(difference)]
        assert finite.size > 0, name
        rms = float(np.sqrt(np.mean(finite**2)))
        assert np.isfinite(rms), name
        lines.append(f"| {name} | {rms:.9g} |")
    path = Path("docs/reproducibility.md")
    original, _, previous = path.read_text().partition("\n## ERA5 initial-condition comparison")
    _, next_heading, following = previous.partition("\n## ")
    path.write_text(original.rstrip() + "\n\n" + "\n".join(lines) + "\n" + next_heading + following)


def test_negative_analysis_humidity_is_clipped_at_zero(store: xr.Dataset, statics: SampleStatics):
    humidity = store["specific_humidity"].copy(deep=True)
    # ERA5 holds a few values like this one; the edit check refuses any below zero.
    humidity[1, 0, 360, 0] = -1.3e-6
    built = build_input(INIT, store.assign(specific_humidity=humidity), 1, statics=statics)
    reference = build_input(INIT, store, 1, statics=statics)
    inputs = built["specific_humidity"].isel(time=slice(0, 2))
    assert float(inputs.min()) == 0
    changed = inputs != reference["specific_humidity"].isel(time=slice(0, 2))
    assert int(changed.sum()) == 1
