"""The forecast commands, with a stand-in forecast step in place of the model.

The model weights are not a test fixture. The stand-in consumes the member key
and the coldest input sea surface temperature, so a changed seed or an edited input
still changes the forecast.
"""

import hashlib
import json
import shutil
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PIL import Image

from stormfork.downloads import HRES_SAMPLE, IBTRACS, OFFLINE_ENV, WEIGHTS
from stormfork.errors import UserError
from stormfork.export.manifest import HINDCAST_LIMIT, ExperimentMetadata, verify_manifest
from stormfork.export.quantise import FILE_NAMES, quantise
from stormfork.grid import LAT, LEVELS_HPA, LON
from stormfork.scenarios import Scenario
from tests.test_cli import invoke
from tests.test_forks import scenario
from tests.test_reproduction import published_tracks

pytest.importorskip("weathernext")

import jax
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn

from stormfork import cli_forecast, cli_rerun
from stormfork.ic.era5 import ATMOSPHERE, SURFACE, TARGETS
from stormfork.run import runner

if TYPE_CHECKING:
    from collections.abc import Callable

zeros_like = cast("Callable[[xr.Dataset], xr.Dataset]", getattr(xr, "zeros_like"))  # noqa: B009

INIT = "2024-10-07T00:00"
MILTON = "2024279N21265"
SST = "sea_surface_temperature"
MSLP = "mean_sea_level_pressure"
SIX_HOURS = np.timedelta64(6, "h").astype("timedelta64[ns]")
PLAUSIBLE = {
    "temperature": 250.0,
    "geopotential": 50000.0,
    "specific_humidity": 0.005,
    "2m_temperature": 290.0,
    "mean_sea_level_pressure": 101000.0,
    "sea_surface_temperature": 300.0,
}


class NetcdfWritable(Protocol):
    def to_netcdf(self, path: Path, *, engine: Literal["h5netcdf"]) -> None: ...


def write_netcdf(dataset: NetcdfWritable, path: Path) -> None:
    dataset.to_netcdf(path, engine="h5netcdf")


def sample_dataset(first_frame: str = "2024-10-06T18") -> xr.Dataset:
    """Two input frames and one target frame of constant fields in the sample layout."""
    frames = 3
    surface_shape = (1, frames, LAT.size, LON.size)
    fields: dict[str, xr.DataArray] = {}
    for name in ATMOSPHERE:
        values = np.full((1, frames, len(LEVELS_HPA), LAT.size, LON.size), PLAUSIBLE.get(name, 1.0))
        fields[name] = xr.DataArray(
            values.astype(np.float32), dims=("batch", "time", "level", "lat", "lon")
        )
    for name in TARGETS:
        if name not in ATMOSPHERE:
            value = PLAUSIBLE.get(name, 1.0) if name in SURFACE else np.nan
            fields[name] = xr.DataArray(
                np.full(surface_shape, value, dtype=np.float32),
                dims=("batch", "time", "lat", "lon"),
            )
    for name in ("geopotential_at_surface", "land_sea_mask"):
        fields[name] = xr.DataArray(
            np.zeros((LAT.size, LON.size), dtype=np.float32), dims=("lat", "lon")
        )
    lead = np.arange(frames) * SIX_HOURS
    dataset = xr.Dataset(
        fields,
        coords={
            "time": lead,
            "level": np.array(LEVELS_HPA, dtype=np.int32),
            "lat": LAT.astype(np.float32),
            "lon": LON.astype(np.float32),
        },
    )
    dataset.coords["datetime"] = (
        ("batch", "time"),
        (np.datetime64(first_frame, "ns") + lead)[None],
    )
    return dataset


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sample") / "sample.nc"
    write_netcdf(sample_dataset(), path)
    return path


@pytest.fixture
def ibtracs(workdir: Path) -> Path:
    """Milton's IBTrACS position at the init and six hours before, in the NetCDF layout."""
    times = np.datetime64(INIT, "ns") - np.array([1, 0]) * SIX_HOURS
    path = workdir / "ibtracs.nc"
    write_netcdf(
        xr.Dataset(
            {
                "time": (("storm", "date_time"), times[None]),
                "sid": ("storm", [MILTON]),
                "lat": (("storm", "date_time"), [[22.4, 22.5]]),
                "lon": (("storm", "date_time"), [[-93.0, -93.4]]),
                "usa_pres": (("storm", "date_time"), [[980.0, 975.0]]),
                "usa_wind": (("storm", "date_time"), [[60.0, 65.0]]),
                "usa_rmw": (("storm", "date_time"), [[20.0, 20.0]]),
            }
        ),
        path,
    )
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.npz"
    path.write_bytes(b"stand-in")
    return path


@pytest.fixture
def workdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run from a scratch directory so reported paths are short and relative."""
    shutil.copytree(Path(__file__).parent.parent / "experiments", tmp_path / "experiments")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def stand_in_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def predictor_fn(config: fgn.PredictorConfig, ckpt: fgn.CheckPoint) -> runner.PredictorFn:
        del config, ckpt

        def predict(
            rng: "jax.Array",
            inputs: xr.Dataset,
            targets_template: xr.Dataset,
            forcings: xr.Dataset,
        ) -> xr.Dataset:
            del forcings
            # A small key term keeps the two-kelvin edit within float32 precision.
            key = float(np.asarray(rng).sum() % 1000)
            predicted = zeros_like(targets_template) + key + float(inputs[SST].min())
            # Like the model, return jax arrays inside the dataset.
            return predicted.copy(
                data={
                    name: jax.device_put(field.data) for name, field in predicted.data_vars.items()
                }
            )

        return predict

    def load(source: BinaryIO, typ: type[fgn.CheckPoint]) -> fgn.CheckPoint:
        assert source.read() == b"stand-in"
        return typ(params={}, description="stand-in", license="none")

    monkeypatch.setattr(runner, "predictor_fn", predictor_fn)
    monkeypatch.setattr(checkpoint, "load", load)


def fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str) -> str:
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, *args)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    return "".join(captured.err.splitlines(keepends=True)[-2:])


def assert_laid_out_for_every_step(built: xr.Dataset, sample: xr.Dataset) -> None:
    """The sample's two input frames, then NaN targets for the longest rollout."""
    assert built.sizes["time"] == 2 + 40
    inputs = slice(0, 2)
    for name in ATMOSPHERE + SURFACE:
        np.testing.assert_array_equal(
            built[name].isel(time=inputs).values, sample[name].isel(time=inputs).values
        )
    for name in TARGETS:
        assert np.isnan(built[name].isel(time=slice(2, None)).values).all()
    np.testing.assert_array_equal(
        built["datetime"].to_numpy()[0],
        np.datetime64(INIT, "ns") + np.arange(-1, 41) * SIX_HOURS,
    )
    for name in ("geopotential_at_surface", "land_sea_mask"):
        assert built[name].identical(sample[name])


def test_ic_build_lays_the_hres_sample_out_for_every_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
):
    invoke(
        monkeypatch, "ic", "build", "milton_noise_1pct", "--init", INIT, "--out", "run/input.nc",
        "--sample", str(sample),
    )  # fmt: skip
    assert capsys.readouterr().out == "model input written to run/input.nc\n"
    built = xr.load_dataset(workdir / "run/input.nc")
    assert_laid_out_for_every_step(built, xr.load_dataset(sample))
    # The sample's own target frame would allow one step.
    arguments = Namespace(runtime=runner.default_backend(), steps=40)
    cli_forecast.check_rollout(built, arguments)


@pytest.mark.parametrize("steps", [1, 8, 40])
def test_a_rollout_of_fewer_steps_than_the_input_holds_reads_its_input_frames(
    sample: Path, steps: int
):
    built = cli_forecast.initial_conditions(scenario(), datetime.fromisoformat(INIT), sample)
    task = cli_forecast.model_task(built, steps)
    assert task.targets.sizes["time"] == task.forcings.sizes["time"] == steps
    for name in ATMOSPHERE + SURFACE:
        np.testing.assert_array_equal(
            task.inputs[name].values, built[name].isel(time=slice(0, 2)).values
        )


def test_ic_build_rejects_a_sample_of_another_init(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workdir: Path
):
    later = sample_dataset("2024-10-07T00")
    write_netcdf(later[[SST]], workdir / "later.nc")
    write_netcdf(later[["land_sea_mask"]], workdir / "other.nc")
    command = (
        "ic", "build", "milton_noise_1pct", "--init", INIT, "--out", "input.nc", "--sample",
    )  # fmt: skip
    assert fail(monkeypatch, capsys, *command, "later.nc") == (
        'error: "later.nc" starts at 2024-10-07T06:00Z\n'
        "next: pass the HRES sample for 2024-10-07T00:00Z\n"
    )
    assert fail(monkeypatch, capsys, *command, "other.nc") == (
        'error: "other.nc" has no datetime coordinate, so it is not an HRES sample\n'
        "next: pass the HRES sample file to --sample\n"
    )
    assert not (workdir / "input.nc").exists()


def test_ic_build_reads_the_cached_sample_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    offline_cache: Path,
):
    shutil.copy(sample, offline_cache / HRES_SAMPLE.filename)
    invoke(
        monkeypatch, "ic", "build", "milton_noise_1pct", "--init", INIT, "--out", "input.nc"
    )  # fmt: skip
    assert capsys.readouterr().out == "model input written to input.nc\n"
    assert_laid_out_for_every_step(xr.load_dataset(workdir / "input.nc"), xr.load_dataset(sample))


def test_ic_build_needs_an_existing_sample(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    command = ("ic", "build", "milton_noise_1pct", "--init", INIT, "--out", "input.nc")
    assert fail(monkeypatch, capsys, *command) == (
        f"error: no cached HRES sample, and {OFFLINE_ENV} is set\n"
        "next: run stormfork fetch, or pass a local file to --sample\n"
    )
    assert fail(monkeypatch, capsys, *command, "--sample", "absent.nc") == (
        'error: HRES sample file "absent.nc" does not exist\n'
        "next: pass an existing file to --sample\n"
    )


@pytest.mark.parametrize("source", ["era5", "ifs_open_hybrid"])
def test_other_sources_are_built_from_the_remote_stores_with_the_sample_statics(
    monkeypatch: pytest.MonkeyPatch, source: str
):
    era5, ifs, statics, built = xr.Dataset(), xr.Dataset(), object(), xr.Dataset()
    opened: list[str] = []
    init = datetime.fromisoformat(INIT)

    def open_zarr(store: str, **options: object) -> xr.Dataset:
        opened.append(store)
        assert options["storage_options"] == {"token": "anon"}
        # Lazy indexing reads two hours; dask chunks would map the whole archive first.
        assert "chunks" in options
        assert options["chunks"] is None
        return era5

    def build_input(time: datetime, store: xr.Dataset, *, statics: object) -> xr.Dataset:
        assert (time, store, statics) == (init, era5, statics)
        assert source == "era5"
        return built

    def build_hybrid(
        time: datetime, atmosphere: xr.Dataset, sst: xr.Dataset, *, statics: object, method: str
    ) -> xr.Dataset:
        assert (time, method) == (init, "subsample")
        assert atmosphere is ifs
        assert sst is era5
        assert statics is not None
        return built

    def load_sample_statics(path: Path) -> object:
        assert path == Path("sample.nc")
        return statics

    def fetch_ifs_store(time: datetime, directory: Path) -> xr.Dataset:
        assert time == init
        assert directory.is_dir()
        return ifs

    monkeypatch.setattr(cli_forecast.xr, "open_zarr", open_zarr)
    monkeypatch.setattr(cli_forecast, "load_sample_statics", load_sample_statics)
    monkeypatch.setattr(cli_forecast, "build_input", build_input)
    monkeypatch.setattr(cli_forecast, "build_hybrid", build_hybrid)
    monkeypatch.setattr(cli_forecast, "fetch_ifs_store", fetch_ifs_store)
    recipe = scenario(kind="hindcast", edit=None, source=source)
    with pytest.raises(UserError, match=f"{OFFLINE_ENV} is set") as raised:
        cli_forecast.initial_conditions(recipe, init, Path("sample.nc"))
    assert cli_forecast.ARCO_ERA5 in raised.value.what
    assert raised.value.next_step == f"unset {OFFLINE_ENV} and retry"
    assert opened == []
    monkeypatch.delenv(OFFLINE_ENV)
    assert cli_forecast.initial_conditions(recipe, init, Path("sample.nc")) is built
    assert opened == [cli_forecast.ARCO_ERA5]


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
def test_run_writes_the_forecasts_atlases_and_metadata_export_reads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    weights: Path,
):
    out = workdir / "run"
    out.mkdir()
    (out / "input.nc").symlink_to(sample)
    invoke(
        monkeypatch, "run", "milton_gulf_minus2k", "--out", "run", "--weights", str(weights),
        "--ibtracs", "ibtracs.nc", "--members", "2", "--steps", "1",
    )  # fmt: skip
    captured = capsys.readouterr()
    assert captured.out == (
        "baseline, fork, control forecasts, atlases and metadata.json written to run\n"
    )
    # The IBTrACS reader adds its own library warnings to stderr.
    assert [line for line in captured.err.splitlines() if "rolling out" in line] == [
        f"{run}: rolling out 2 members for 1 steps" for run in ("baseline", "fork", "control")
    ]
    baseline, fork, control = (
        xr.load_dataset(out / run / "forecast.nc") for run in ("baseline", "fork", "control")
    )
    assert baseline.sizes["member"] == 2
    assert baseline.sizes["time"] == 1
    assert "temperature" not in baseline
    assert set(TARGETS) - set(ATMOSPHERE) <= set(baseline.data_vars)
    assert baseline.geopotential.level.to_numpy().tolist() == [500]
    assert baseline.time.to_numpy()[0] == SIX_HOURS
    pressure = "mean_sea_level_pressure"
    members = baseline[pressure].max(("batch", "time", "lat", "lon")).to_numpy()
    assert members[0] != members[1]
    cooled = float((fork[pressure] - baseline[pressure]).max())
    assert cooled == pytest.approx(-2, abs=0.05)
    assert not baseline.identical(control)
    metadata = ExperimentMetadata.model_validate_json((out / "metadata.json").read_bytes())
    assert metadata.id == "milton_gulf_minus2k"
    assert (metadata.members, metadata.steps) == (2, 1)
    assert metadata.init.storm.ibtracs_sid == MILTON
    assert (metadata.init.storm.lat, metadata.init.storm.lon) == pytest.approx((22.5, 360 - 93.4))
    assert metadata.init.storm.first_record_time == datetime.fromisoformat("2024-10-06T18:00")
    # The SST edit has no taper, so the ratio is read around the storm's initial position.
    assert metadata.diagnostics.adjustment_ratio == pytest.approx(1, rel=1e-3)
    assert metadata.diagnostics.core_land_fraction is None
    assert metadata.diagnostics.bounds_check == "passed"
    assert metadata.model.checkpoint == "weights.npz"
    assert metadata.model.md5 == hashlib.md5(b"stand-in", usedforsecurity=False).hexdigest()
    assert metadata.runtime.kind == "cpu"
    assert metadata.attribution == ["none", HRES_SAMPLE.credit]
    for run in ("baseline", "fork", "control"):
        for file in FILE_NAMES:
            with Image.open(out / run / f"{file}.png") as atlas:
                assert atlas.size == (1805, 1629)
            assert (out / "preview" / f"{run}_{file}_f0.png").is_file()
    with Image.open(out / "baseline/mslp.png") as atlas:
        pixels = np.asarray(atlas)
    np.testing.assert_array_equal(
        pixels[:181, :360], quantise(np.full((181, 360), 101000.0), "mslp")
    )
    member_zero = baseline[pressure].isel(member=0, batch=0, time=0).to_numpy()
    np.testing.assert_array_equal(pixels[:181, 361:721], quantise(member_zero, "mslp"))
    assert not pixels[181:, :].any()


def write_hindcast(workdir: Path, *inits: str) -> None:
    recipe = {
        "id": "hindcast_test",
        "kind": "hindcast",
        "storm": MILTON,
        "init": list(inits),
        "source": "era5",
        "seeds": {"baseline": 0, "control": 1000},
        "limits": [HINDCAST_LIMIT],
    }
    (workdir / "experiments/hindcast_test.yaml").write_text(json.dumps(recipe))


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
def test_run_rolls_out_a_hindcast_baseline_at_its_input_init(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    weights: Path,
):
    write_hindcast(workdir, "2024-10-04T00:00Z", INIT)
    out = workdir / "run"
    out.mkdir()
    (out / "input.nc").symlink_to(sample)
    invoke(
        monkeypatch, "run", "hindcast_test", "--out", "run", "--weights", str(weights),
        "--ibtracs", "ibtracs.nc", "--members", "1", "--steps", "1",
    )  # fmt: skip
    assert (
        capsys.readouterr().out == "baseline forecasts, atlases and metadata.json written to run\n"
    )
    assert sorted(path.name for path in out.iterdir()) == [
        "baseline", "input.nc", "metadata.json", "preview",
    ]  # fmt: skip
    metadata = ExperimentMetadata.model_validate_json((out / "metadata.json").read_bytes())
    assert metadata.kind == "hindcast"
    assert (metadata.id, metadata.parent_id) == ("hindcast_test-2024100700", "hindcast_test")
    assert metadata.init.time == datetime.fromisoformat(INIT)
    assert (metadata.init.storm.lat, metadata.init.storm.lon) == pytest.approx((22.5, 360 - 93.4))
    assert metadata.perturbation.model_dump() == {"type": "none", "params": {}}
    assert metadata.diagnostics.adjustment_ratio is None
    assert metadata.diagnostics.bounds_check is None


@pytest.mark.usefixtures("ibtracs")
def test_run_refuses_an_input_at_another_init(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    weights: Path,
):
    out = workdir / "run"
    out.mkdir()
    write_netcdf(sample_dataset(first_frame="2024-10-06T06"), out / "input.nc")
    write_hindcast(workdir, INIT)
    assert fail(
        monkeypatch, capsys, "run", "hindcast_test", "--out", "run", "--weights", str(weights),
        "--ibtracs", "ibtracs.nc", "--members", "1", "--steps", "1",
    ) == (
        "error: run/input.nc starts at 2024-10-06T12:00Z, not an init of hindcast_test\n"
        "next: rebuild it with stormfork ic build hindcast_test --init <an init>\n"
    )  # fmt: skip


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
def test_run_rolls_out_a_hindcast_init_before_the_storm_forms(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    weights: Path,
):
    out = workdir / "run"
    out.mkdir()
    write_netcdf(sample_dataset(first_frame="2024-10-06T06"), out / "input.nc")
    write_hindcast(workdir, "2024-10-06T12:00Z")
    invoke(
        monkeypatch, "run", "hindcast_test", "--out", "run", "--weights", str(weights),
        "--ibtracs", "ibtracs.nc", "--members", "1", "--steps", "1",
    )  # fmt: skip
    assert capsys.readouterr().out.startswith("baseline forecasts")
    metadata = ExperimentMetadata.model_validate_json((out / "metadata.json").read_bytes())
    assert metadata.id == "hindcast_test-2024100612"
    assert (metadata.init.storm.lat, metadata.init.storm.lon) == (None, None)
    assert metadata.init.storm.first_record_time == datetime.fromisoformat("2024-10-06T18:00")


@pytest.mark.usefixtures("ibtracs")
def test_run_refuses_a_storm_ibtracs_does_not_hold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    weights: Path,
):
    out = workdir / "run"
    out.mkdir()
    (out / "input.nc").symlink_to(sample)
    recipe = (workdir / "experiments/milton_noise_1pct.yaml").read_text()
    (workdir / "experiments/milton_noise_1pct.yaml").write_text(
        recipe.replace(MILTON, "2024001N00000")
    )
    assert fail(
        monkeypatch, capsys, "run", "milton_noise_1pct", "--out", "run", "--weights",
        str(weights), "--ibtracs", "ibtracs.nc", "--members", "1", "--steps", "1",
    ) == (
        "error: storm 2024001N00000 has no IBTrACS position at 2024-10-07T00:00Z\n"
        "next: check the storm id and pick an init from stormfork scenarios list\n"
    )  # fmt: skip


@pytest.mark.usefixtures("ibtracs")
def test_run_refuses_what_it_cannot_roll_out(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    weights: Path,
):
    out = workdir / "run"
    command = (
        "run", "milton_noise_1pct", "--out", "run", "--weights", "weights.npz",
        "--ibtracs", "ibtracs.nc",
    )  # fmt: skip
    assert fail(monkeypatch, capsys, *command, "--steps", "0").startswith(
        "error: --members and --steps must be at least 1\n"
    )
    assert fail(monkeypatch, capsys, *command, "--steps", "41") == (
        "error: an atlas holds the analysis and at most 40 six-hour steps\n"
        "next: pass --steps between 1 and 40\n"
    )
    assert weights.is_file()
    assert fail(monkeypatch, capsys, *command[:-4], "--out", "r") == (
        'error: "r" has no input.nc\n'
        "next: run stormfork ic build milton_noise_1pct with --out r/input.nc\n"
    )
    out.mkdir()
    (out / "input.nc").symlink_to(sample)
    assert fail(monkeypatch, capsys, *command[:-4]) == (
        f"error: no cached model weights, and {OFFLINE_ENV} is set\n"
        "next: run stormfork fetch, or pass a local file to --weights\n"
    )
    assert fail(monkeypatch, capsys, *command[:-2]) == (
        f"error: no cached IBTrACS, and {OFFLINE_ENV} is set\n"
        "next: run stormfork fetch, or pass a local file to --ibtracs\n"
    )
    assert fail(monkeypatch, capsys, *command, "--runtime", "gpu") == (
        "error: --runtime gpu was requested, but jax is running on cpu\n"
        "next: pass --runtime cpu, or install jax for gpu\n"
    )
    assert fail(monkeypatch, capsys, *command) == (
        "error: the model input holds 1 six-hour steps, fewer than the 40 requested\n"
        "next: pass --steps 1, or build a longer input\n"
    )
    assert not (out / "baseline").exists()


def write_sweep(workdir: Path) -> None:
    recipe = {
        "id": "sweep_test",
        "kind": "sweep",
        "storm": MILTON,
        "init": INIT,
        "source": "hres_sample",
        "edit": {"type": "relocate", "params": {"north_km": 0}},
        "sweep": {"param": "east_km", "values": [-150, 0, 150]},
        "taper": {"centre": [22.5, -93.4], "radius_km": 400.0},
        "seeds": {"baseline": 0, "control": 1000},
    }
    (workdir / "experiments/sweep_test.yaml").write_text(json.dumps(recipe))


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
def test_run_rolls_out_a_sweep_with_its_null_fork_and_each_stop_the_edit_check_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    weights: Path,
):
    write_sweep(workdir)
    model_input = sample_dataset()
    # Land east of 269 E lies in the core moved 150 km east, and outside the unmoved one.
    model_input.land_sea_mask.loc[{"lat": slice(15, 30), "lon": slice(269, 275)}] = 1
    out = workdir / "run"
    out.mkdir()
    write_netcdf(model_input, out / "input.nc")
    invoke(
        monkeypatch, "run", "sweep_test", "--out", "run", "--weights", str(weights),
        "--ibtracs", "ibtracs.nc", "--members", "2", "--steps", "1",
    )  # fmt: skip
    assert " ".join(capsys.readouterr().out.split()) == (
        "baseline, fork, stops/east_km_-150, control forecasts, atlases and metadata.json "
        "written to run"
    )
    baseline, fork = (xr.load_dataset(out / run / "forecast.nc") for run in ("baseline", "fork"))
    assert baseline.identical(fork)
    stop = out / "stops/east_km_-150"
    assert sorted(path.name for path in stop.iterdir()) == ["forecast.nc", "mslp.png"]
    with Image.open(stop / "mslp.png") as atlas:
        assert atlas.size == (1805, 1629)
    assert not (out / "stops/east_km_150").exists()
    stops = json.loads((out / "stops.json").read_text())
    assert stops[0] == {
        "run": "stops/east_km_-150",
        "param": "east_km",
        "value": -150,
        "adjustment_ratio": pytest.approx(1),
        "core_land_fraction": 0,
        "refused": None,
    }
    assert stops[1]["run"] == "stops/east_km_150"
    assert stops[1]["adjustment_ratio"] is None
    assert stops[1]["refused"].startswith("land_sea_mask: core land fraction must be < 0.2")
    metadata = ExperimentMetadata.model_validate_json((out / "metadata.json").read_bytes())
    assert metadata.kind == "sweep"
    assert metadata.diagnostics.adjustment_ratio == pytest.approx(1)
    assert metadata.diagnostics.core_land_fraction == 0
    assert metadata.diagnostics.bounds_check == "passed"


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
@pytest.mark.parametrize("hero", [("2", 3 + 4), ("1", 3 + 5)], ids=["same", "other_members"])
def test_a_sweep_stop_equal_to_a_published_fork_reuses_its_forecast(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    weights: Path,
    hero: tuple[str, int],
):
    hero_members, rollouts = hero
    write_sweep(workdir)
    # Equal to the sweep's +150 km stop in content, under another name and title.
    recipe = {
        "id": "hero_east_150",
        "title": "Moved 150 km east",
        "kind": "fork",
        "storm": MILTON,
        "init": INIT,
        "source": "hres_sample",
        "edit": {"type": "relocate", "params": {"east_km": 150, "north_km": 0}},
        "taper": {"centre": [22.5, -93.4], "radius_km": 400.0},
        "seeds": {"baseline": 0, "control": 1000},
    }
    (workdir / "experiments/hero_east_150.yaml").write_text(json.dumps(recipe))
    calls: list[int] = []
    rollout = runner.rollout

    def counted(*args: Any, **kwargs: Any) -> xr.Dataset:  # noqa: ANN401 - passes the model's arguments through
        calls.append(len(calls))
        return rollout(*args, **kwargs)

    monkeypatch.setattr(runner, "rollout", counted)
    for experiment, members in (("hero_east_150", hero_members), ("sweep_test", "2")):
        (workdir / experiment).mkdir()
        write_netcdf(sample_dataset(), workdir / experiment / "input.nc")
        invoke(
            monkeypatch, "run", experiment, "--out", experiment, "--weights", str(weights),
            "--ibtracs", "ibtracs.nc", "--members", members, "--steps", "1",
        )  # fmt: skip
    assert len(calls) == rollouts
    reused = "stops/east_km_150: reusing the forecast in hero_east_150/fork"
    assert (reused in capsys.readouterr().err) == (hero_members == "2")
    stop = workdir / "sweep_test/stops/east_km_150"
    hero_fork = xr.load_dataset(workdir / "hero_east_150/fork/forecast.nc")
    assert xr.load_dataset(stop / "forecast.nc").identical(hero_fork) == (hero_members == "2")
    assert (stop / "mslp.png").is_file()
    stops = json.loads((workdir / "sweep_test/stops.json").read_text())
    assert stops[1]["run"] == "stops/east_km_150"
    assert stops[1]["adjustment_ratio"] == pytest.approx(1)


@pytest.mark.usefixtures("stand_in_model", "ibtracs")
def test_rerun_of_a_recipe_tracks_each_run_and_exports_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    sample: Path,
    weights: Path,
):
    tracked: list[tuple[int, np.datetime64, str | None, tuple[float, float] | None]] = []
    tracks = iter(published_tracks().values())

    def track(
        fields: xr.Dataset,
        ibtracs: xr.Dataset,
        init_time: np.datetime64,
        *,
        target_track_id: str | None,
        moved_initial_position: tuple[float, float] | None,
    ) -> pd.DataFrame:
        assert ibtracs.sid.to_numpy().tolist() == [MILTON]
        # The tracker indexes in ways only NumPy-backed fields support.
        assert all(isinstance(field.data, np.ndarray) for field in fields.data_vars.values())
        tracked.append((fields.sizes["member"], init_time, target_track_id, moved_initial_position))
        return next(tracks)

    monkeypatch.setattr(cli_rerun, "track", track)
    recipe = workdir / "mine.yaml"
    recipe.write_text((workdir / "experiments/milton_south_75km.yaml").read_text())
    invoke(
        monkeypatch, "rerun", "mine.yaml", "--out", "out", "--steps", "1",
        "--sample", str(sample), "--weights", str(weights), "--ibtracs", "ibtracs.nc",
    )  # fmt: skip
    assert capsys.readouterr().out == (
        "manifest written to out/manifest.json\n"
        "step: 4\n"
        "fork_sep_km: 234.6\n"
        "control_sep_km: 46.9\n"
        "paired_p: 0.0039\n"
    )
    assert [call[:3] for call in tracked] == [(1, np.datetime64(INIT), MILTON)] * 3
    baseline_position, moved, control_position = (call[3] for call in tracked)
    assert baseline_position is None
    assert control_position is None
    assert moved == pytest.approx((22.5 - 75 / 111.195, -93.4), abs=1e-3)
    out = workdir / "out"
    manifest = verify_manifest(out)
    assert manifest.id == "milton_south_75km"
    assert (manifest.members, manifest.steps) == (1, 1)
    # The stand-in model moves both runs to the same constant, and the edit keeps
    # the uniform initial pressure, so the two tendencies are equal.
    assert manifest.diagnostics.adjustment_ratio == pytest.approx(1)
    assert set(manifest.runs) == {"baseline", "fork", "control"}
    for run in ("baseline", "fork", "control"):
        assert (out / run / "forecast.nc").is_file()


@pytest.mark.parametrize("flag", ["sample", "weights", "ibtracs"])
def test_rerun_fails_fast_offline_on_each_uncached_input(tmp_path: Path, flag: str):
    present = tmp_path / "present"
    present.write_bytes(b"")
    files = dict.fromkeys(("sample", "weights", "ibtracs"), present) | {flag: None}
    with pytest.raises(UserError, match=f"{OFFLINE_ENV} is set") as raised:
        cli_rerun.run_forecasts(scenario(), Namespace(**files), tmp_path / "out")
    assert raised.value.next_step == f"run stormfork fetch, or pass a local file to --{flag}"
    assert not (tmp_path / "out").exists()


def test_rerun_takes_every_input_from_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample: Path, offline_cache: Path
):
    for source in (HRES_SAMPLE, WEIGHTS, IBTRACS):
        shutil.copy(sample, offline_cache / source.filename)
    read: list[Path] = []

    def initial_conditions(recipe: Scenario, init: datetime, path: Path) -> xr.Dataset:
        del recipe, init
        read.append(path)
        msg = "stop"
        raise UserError(msg, msg, 2)

    monkeypatch.setattr(cli_forecast, "initial_conditions", initial_conditions)
    arguments = Namespace(sample=None, weights=None, ibtracs=None)
    with pytest.raises(UserError, match="stop"):
        cli_rerun.run_forecasts(scenario(), arguments, tmp_path / "out")
    assert read == [offline_cache / HRES_SAMPLE.filename]


def test_rerun_tracks_each_sweep_stop_from_where_its_edit_moved_the_storm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample: Path, offline_cache: Path
):
    for source in (HRES_SAMPLE, WEIGHTS, IBTRACS):
        shutil.copy(sample, offline_cache / source.filename)
    sweep = scenario(
        kind="sweep",
        edit={"type": "relocate", "params": {"north_km": 0}},
        sweep={"param": "east_km", "values": [0, 150]},
        taper={"centre": [22.5, -93.4], "radius_km": 400.0},
    )
    runs = ["baseline", "fork", "stops/east_km_150", "control"]
    for run in runs:
        (tmp_path / run).mkdir(parents=True)
    positions: dict[str, tuple[float, float] | None] = {}

    def track(
        fields: xr.Dataset,
        *_: object,
        moved_initial_position: tuple[float, float] | None,
        **__: object,
    ) -> pd.DataFrame:
        positions[str(fields.attrs["run"])] = moved_initial_position
        return pd.DataFrame()

    def empty(*_: object) -> xr.Dataset:
        return xr.Dataset()

    def forecast_runs(*_: object) -> list[str]:
        return runs

    def forecast(path: Path) -> xr.Dataset:
        return xr.Dataset(attrs={"run": path.parent.name})

    monkeypatch.setattr(cli_forecast, "initial_conditions", empty)
    monkeypatch.setattr(cli_forecast, "forecast_run_directory", forecast_runs)
    monkeypatch.setattr(cli_rerun, "load_dataset", forecast)
    monkeypatch.setattr(cli_rerun, "track", track)
    arguments = Namespace(sample=None, weights=None, ibtracs=None)
    cli_rerun.run_forecasts(sweep, arguments, tmp_path)
    assert positions["baseline"] is None
    assert positions["fork"] == pytest.approx((22.5, -93.4))
    # 150 km is 1.460 degrees of longitude at 22.5 N.
    assert positions["east_km_150"] == pytest.approx((22.5, -91.940), abs=1e-3)
    assert positions["control"] is None


def test_the_model_record_names_faster_weathernext_when_it_ran_the_model(
    monkeypatch: pytest.MonkeyPatch, weights: Path
):
    versions = {"stormfork": "1.0", "weathernext": "0.3", "faster-weathernext": "0.2"}
    monkeypatch.setattr(cli_forecast, "version", versions.__getitem__)
    ckpt = fgn.CheckPoint(params={}, description="stand-in", license="none")
    assert cli_forecast.model_info(ckpt, weights).code_version == "stormfork 1.0, weathernext 0.3"
    monkeypatch.setattr(runner, "uses_faster_weathernext", lambda: True)
    assert cli_forecast.model_info(ckpt, weights).code_version == (
        "stormfork 1.0, weathernext 0.3, faster-weathernext 0.2"
    )
