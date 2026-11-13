"""Build model input and roll out an experiment's baseline, fork and control.

This module imports the model stack, so the command line loads it only after
the stack check has passed. Weights, the HRES sample and IBTrACS come from the
download cache unless the caller passes local files; the ERA5 and IFS stores
are read remotely.
"""

import hashlib
import json
import shutil
import sys
from argparse import Namespace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
import xarray as xr
from weathernext.utils import checkpoint
from weathernext.utils.data_utils import extract_inputs_targets_forcings
from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name
from weathernext.weathernext2 import fgn

from stormfork.cli_data import load_dataset, parse_time
from stormfork.cli_export import METADATA_FILE, STOPS_FILE
from stormfork.cli_model import MAX_STEPS, input_file, known_scenario
from stormfork.downloads import HRES_SAMPLE, IBTRACS, WEIGHTS, require_online
from stormfork.errors import UserError
from stormfork.export.atlas import export_run, pack_atlas
from stormfork.export.frames import ATLAS_VARIABLES, Z500_LEVEL_HPA, atlas_fields
from stormfork.export.manifest import Diagnostics, ExperimentMetadata, ModelInfo, RuntimeInfo
from stormfork.export.metadata import experiment_metadata
from stormfork.export.preview import write_grayscale
from stormfork.export.quantise import field_metadata, quantise
from stormfork.grid import STEP
from stormfork.ic.era5 import build_input, frame_times, lay_out
from stormfork.ic.ifs_open import build_hybrid, fetch_ifs_store
from stormfork.ic.layout import load_sample_statics
from stormfork.run import runner
from stormfork.run.forks import (
    STOPS_DIR,
    edit_forks,
    first_step_adjustment_ratio,
    first_step_pressure,
    fork_recipes,
    same_forecast,
    sweep_stops,
)
from stormfork.scenarios import Scenario, format_time, load_scenarios
from stormfork.track import StormStart, storm_start

if TYPE_CHECKING:
    from numpy.typing import NDArray

MODEL_CONFIG = "weathernext2/configs/WeatherNextCyclones_Mini"
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
INPUT_FILE = "input.nc"
FORECAST_FILE = "forecast.nc"
Z500_NAME = ATLAS_VARIABLES["z500"]
INPUT_FRAMES = 2
STEP_HOURS = STEP.total_seconds() / 3600


class ModelTask(NamedTuple):
    """The model config and the input, target and forcing frames every run of an init shares."""

    config: fgn.PredictorConfig
    inputs: xr.Dataset
    targets: xr.Dataset
    forcings: xr.Dataset


class NetcdfWritable(Protocol):
    """The one overload of xarray's writer used here; the others are untyped."""

    def to_netcdf(self, path: Path) -> object:
        """Write the dataset to a NetCDF file."""
        ...


def write_netcdf(dataset: NetcdfWritable, path: Path) -> None:
    """Write a dataset, creating the directories above it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)


def initial_conditions(scenario: Scenario, init: datetime, sample: Path) -> xr.Dataset:
    """Build the model input for one init from the experiment's source.

    The HRES sample gives its two input frames and the other sources their
    analyses; every source takes its terrain and land mask from the sample, so
    every experiment shares the same statics. The sample's own target frame
    ends six hours on, so its inputs are laid out like the others, with NaN
    targets for the longest rollout.

    Raises:
        UserError: The file is not an HRES sample, or starts at another time than the init.
    """
    if scenario.source == "hres_sample":
        dataset = load_dataset(sample)
        if "datetime" not in dataset.coords:
            msg = f'"{sample}" has no datetime coordinate, so it is not an HRES sample'
            raise UserError(msg, "pass the HRES sample file to --sample", 2)
        sample_init = dataset["datetime"].values[0, INPUT_FRAMES - 1]
        if sample_init != np.datetime64(init):
            msg = f'"{sample}" starts at {np.datetime_as_string(sample_init, unit="m")}Z'
            raise UserError(msg, f"pass the HRES sample for {format_time(init)}", 2)
        frames = dataset.isel(batch=0, time=slice(0, INPUT_FRAMES))
        return lay_out(frames, frame_times(init, MAX_STEPS), statics=load_sample_statics(sample))
    statics = load_sample_statics(sample)
    require_online(ARCO_ERA5)
    # Dask chunking would build a graph over every hourly chunk since 1940 for
    # each of the store's variables, gigabytes before a byte of data is read.
    with xr.open_zarr(
        ARCO_ERA5, storage_options={"token": "anon"}, consolidated=True, chunks=None
    ) as era5:
        if scenario.source == "era5":
            return build_input(init, era5, statics=statics)
        with TemporaryDirectory() as downloads:
            ifs = fetch_ifs_store(init, Path(downloads))
            return build_hybrid(init, ifs, era5, statics=statics, method="subsample")


def ic_build_output(arguments: Namespace) -> str:
    """Write the model input for one init of an experiment."""
    scenario = known_scenario(arguments.id)
    sample = input_file(arguments, HRES_SAMPLE)
    dataset = initial_conditions(scenario, parse_time(arguments.init), sample)
    write_netcdf(dataset, arguments.out)
    return f"model input written to {arguments.out}"


def check_rollout(model_input: xr.Dataset, arguments: Namespace) -> None:
    """Check the requested runtime and step count before anything heavy is loaded.

    Raises:
        UserError: jax runs on another backend than the one requested, or the
            input holds fewer target frames than the requested steps.
    """
    runtime: str = arguments.runtime
    steps: int = arguments.steps
    backend = runner.default_backend()
    if backend != runtime:
        msg = f"--runtime {runtime} was requested, but jax is running on {backend}"
        raise UserError(msg, f"pass --runtime {backend}, or install jax for {runtime}", 2)
    available = model_input.sizes["time"] - INPUT_FRAMES
    if steps > available:
        msg = f"the model input holds {available} six-hour steps, fewer than the {steps} requested"
        raise UserError(msg, f"pass --steps {available}, or build a longer input", 2)


def model_task(model_input: xr.Dataset, steps: int) -> ModelTask:
    """Split the model input into the frames the model reads, for the requested steps.

    The extraction counts the input frames back from the input's last frame,
    so frames past the last requested step are dropped first.
    """
    config = get_fiddle_config_by_name(MODEL_CONFIG)
    task = config.task
    inputs, targets, forcings = extract_inputs_targets_forcings(
        model_input.isel(time=slice(0, INPUT_FRAMES + steps)),
        input_variables=task.input_variables,
        target_variables=task.target_variables,
        forcing_variables=task.forcing_variables,
        pressure_levels=task.pressure_levels,
        input_duration=task.input_duration,
        target_lead_times=slice(f"{STEP_HOURS:.0f}h", f"{steps * STEP_HOURS:.0f}h"),
    )
    return ModelTask(config, inputs, targets, forcings)


def forecast_fields(prediction: xr.Dataset) -> xr.Dataset:
    """Keep what the tracker, atlases and adjustment ratio read: surface fields and z500.

    Every other pressure level of every member and step would outgrow a small
    machine's memory long before the forecast is written.
    """
    surface = [name for name, field in prediction.data_vars.items() if "level" not in field.dims]
    return prediction[[*surface, Z500_NAME]].sel(level=[Z500_LEVEL_HPA])


def write_stop_atlas(inputs: xr.Dataset, forecast: xr.Dataset, path: Path) -> None:
    """Write member 0's pressure atlas, the one field a sweep stop publishes."""
    write_grayscale(pack_atlas(quantise(atlas_fields(inputs, forecast)["mslp"], "mslp")), path)


def roll_out(
    runs: dict[str, tuple[xr.Dataset, int]], task: ModelTask, ckpt: fgn.CheckPoint,
    arguments: Namespace, out: Path,
) -> dict[str, xr.Dataset]:  # fmt: skip
    """Roll out each run from its input and seed, write it, and keep member 0's first step.

    Members lie along ``member``. Each run's forecast.nc and atlases are
    written before the next run starts, so only one forecast is in memory; a
    sweep stop writes only its pressure atlas. The arguments carry the member
    and step counts, which check_rollout has checked against the input.
    """
    members: int = arguments.members
    steps: int = arguments.steps
    pressure: dict[str, xr.Dataset] = {}
    for run, (inputs, seed) in runs.items():
        sys.stderr.write(f"{run}: rolling out {members} members for {steps} steps\n")
        forecast = runner.rollout(
            inputs, task.targets, task.forcings, ckpt, task.config, steps, members, seed,
            keep=forecast_fields,
        ).rename(sample="member")  # fmt: skip
        write_netcdf(forecast, out / run / FORECAST_FILE)
        if run.startswith(f"{STOPS_DIR}/"):
            write_stop_atlas(inputs, forecast, out / run / "mslp.png")
        else:
            export_run(atlas_fields(inputs, forecast), out, run)
        pressure[run] = first_step_pressure(inputs, forecast)
    return pressure


def model_info(ckpt: fgn.CheckPoint, weights: Path) -> ModelInfo:
    """Identify the checkpoint by its file and digest, and the code that ran it."""
    with weights.open("rb") as source:
        md5 = hashlib.file_digest(source, lambda: hashlib.md5(usedforsecurity=False))
    code = [f"{package} {version(package)}" for package in ("stormfork", "weathernext")]
    if runner.uses_faster_weathernext():
        code.append(f"faster-weathernext {version('faster-weathernext')}")
    return ModelInfo(
        name=ckpt.description,
        checkpoint=weights.name,
        md5=md5.hexdigest(),
        code_version=", ".join(code),
    )


def input_init(scenario: Scenario, model_input: xr.Dataset, path: Path) -> datetime:
    """Read the init a model input starts from, which must be one of the experiment's.

    Raises:
        UserError: The input has no datetime coordinate or starts at another time.
    """
    rebuild = f"rebuild it with stormfork ic build {scenario.id} --init <an init>"
    if "datetime" not in model_input.coords:
        msg = f"{path} has no datetime coordinate"
        raise UserError(msg, rebuild, 2)
    stamps: NDArray[np.datetime64] = model_input["datetime"].to_numpy()
    init = datetime.fromisoformat(str(stamps[0, INPUT_FRAMES - 1].astype("datetime64[s]")))
    if init not in scenario.init:
        msg = f"{path} starts at {format_time(init)}, not an init of {scenario.id}"
        raise UserError(msg, rebuild, 2)
    return init


def storm_at_init(storms: xr.Dataset, init: datetime, storm: str) -> StormStart:
    """Find the target storm at the init, which the manifest records.

    An init before the storm's first record is valid; the storm then has no position.

    Raises:
        UserError: IBTrACS has no record of the storm, or no position for it
            at an init after its first record.
    """
    try:
        return storm_start(storms, np.datetime64(init), storm)
    except ValueError as error:
        msg = f"storm {storm} has no IBTrACS position at {format_time(init)}"
        raise UserError(
            msg, "check the storm id and pick an init from stormfork scenarios list", 2
        ) from error


def ran_the_same(
    directory: Path, recipe: Scenario, init: datetime, settings: ExperimentMetadata
) -> bool:
    """Whether a run directory holds the recipe's fork from this init, run as settings records."""
    path = directory / METADATA_FILE
    if not (path.is_file() and (directory / "fork" / FORECAST_FILE).is_file()):
        return False
    ran = ExperimentMetadata.model_validate_json(path.read_bytes())
    started = (ran.init.time, ran.init.source, ran.perturbation)
    return started == (init, recipe.source, recipe.edit) and all(
        getattr(ran, name) == getattr(settings, name)
        for name in ("seeds", "members", "steps", "model", "runtime")
    )


def reusable_forks(
    scenario: Scenario, init: datetime, out: Path, settings: ExperimentMetadata
) -> dict[str, Path]:
    """Find the sweep stops another experiment's fork already rolled out, by run.

    A stop is reusable when a published fork recipe equals the stop's in
    content, and that experiment's run directory, beside out, holds its fork
    run from this init with the same seeds, members, steps, model and runtime
    as settings records for this run. Each reusable stop maps to that fork's
    directory. Without published recipes, nothing is reused.
    """
    if scenario.sweep is None:
        return {}
    try:
        published = [other for other in load_scenarios() if other.kind == "fork"]
    except UserError:
        return {}
    reusable: dict[str, Path] = {}
    for run, recipe in fork_recipes(scenario).items():
        for other in published if run != "fork" else []:
            directory = out.parent / other.id
            if same_forecast(recipe, other) and ran_the_same(directory, recipe, init, settings):
                reusable[run] = directory / "fork"
    return reusable


def forecast_run_directory(  # noqa: PLR0913, PLR0917
    scenario: Scenario,
    init: datetime,
    model_input: xr.Dataset,
    weights: Path,
    storms: xr.Dataset,
    arguments: Namespace,
    out: Path,
) -> list[str]:
    """Roll out a fork, sweep or hindcast init and write everything export reads but the tracks.

    Each run gets its forecast.nc, which the track command reads, and member 0's
    atlases and previews; the directory gets the metadata.json that describes
    the run, with the fork's adjustment ratio, core land fraction and passed
    bounds check. A sweep's null value is its fork; every other value is a stop
    with its forecast.nc and pressure atlas, described in stops.json with the
    stops the edit check refused. A stop another experiment's fork already
    rolled out is copied from it, not rolled out again. The fork shares the
    baseline's seed, so only the edit separates them; the control shares the
    baseline's inputs, so only its seed does. A hindcast has no edit, so only
    its baseline is rolled out.

    Returns:
        The run directories written, in roll-out order with copied stops in place.

    Raises:
        UserError: jax runs on another backend than the one requested, the
            input holds fewer target frames than the requested steps, or
            IBTrACS has no position for the storm at an init after its first record.
        PhysicalBoundsError: The fork's edited input leaves the allowed ranges.
        ValueError: The baseline's pressure does not change in the taper annulus.
    """
    check_rollout(model_input, arguments)
    storm = storm_at_init(storms, init, scenario.storm)
    task = model_task(model_input, arguments.steps)
    edited, refused = edit_forks(scenario, task.inputs)
    with weights.open("rb") as source:
        ckpt = checkpoint.load(source, fgn.CheckPoint)
    metadata = experiment_metadata(
        scenario,
        init=init,
        model=model_info(ckpt, weights),
        weights_license=ckpt.license,
        runtime=RuntimeInfo(kind=arguments.runtime, device=runner.device_kind()),
        storm=storm,
        members=arguments.members,
        steps=arguments.steps,
        fields=field_metadata(),
        diagnostics=Diagnostics(),
        created=datetime.now(UTC),
    )
    reused = reusable_forks(scenario, init, out, metadata)
    seeds = scenario.seeds
    runs = {"baseline": (task.inputs, seeds.baseline)}
    # The fork, a sweep's null fork, goes first so its stops follow it.
    forks = sorted(edited.items(), key=lambda item: item[0] != "fork")
    runs.update({run: (fork.inputs, seeds.baseline) for run, fork in forks})
    if scenario.kind != "hindcast":
        runs["control"] = (task.inputs, seeds.control)
    rolled = {run: inputs for run, inputs in runs.items() if run not in reused}
    pressure = roll_out(rolled, task, ckpt, arguments, out)
    for run, source in reused.items():
        sys.stderr.write(f"{run}: reusing the forecast in {source}\n")
        (out / run).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / FORECAST_FILE, out / run / FORECAST_FILE)
        forecast = load_dataset(out / run / FORECAST_FILE)
        write_stop_atlas(edited[run].inputs, forecast, out / run / "mslp.png")
        pressure[run] = first_step_pressure(edited[run].inputs, forecast)
    diagnostics = Diagnostics()
    if "fork" in edited:
        diagnostics = Diagnostics(
            adjustment_ratio=first_step_adjustment_ratio(
                fork_recipes(scenario)["fork"], storm, pressure["fork"], pressure["baseline"]
            ),
            core_land_fraction=edited["fork"].core_land_fraction,
            bounds_check="passed",
        )  # fmt: skip
    if scenario.sweep is not None:
        stops = sweep_stops(scenario, storm, edited, refused, pressure)
        document = [stop.model_dump(mode="json") for stop in stops]
        (out / STOPS_FILE).write_text(json.dumps(document, indent=2) + "\n")
    metadata = metadata.model_copy(
        update={"diagnostics": diagnostics, "created": datetime.now(UTC)}
    )
    (out / METADATA_FILE).write_text(metadata.model_dump_json(indent=2) + "\n")
    return list(runs)


def run_output(arguments: Namespace) -> str:
    """Roll out a fork, a sweep or a hindcast from the model input in its run directory.

    The input's own init is the run's, so a hindcast runs once per built init,
    and each init is its own experiment.

    Raises:
        UserError: The run directory has no model input yet, the input starts
            at another time, or an input file is neither passed nor available
            from the cache.
    """
    scenario = known_scenario(arguments.id)
    out: Path = arguments.out or Path(scenario.id)
    model_input = out / INPUT_FILE
    if not model_input.is_file():
        msg = f'"{out}" has no {INPUT_FILE}'
        build = f"stormfork ic build {scenario.id} with --out {model_input}"
        raise UserError(msg, f"run {build}", 2)
    dataset = load_dataset(model_input)
    init = input_init(scenario, dataset, model_input)
    weights = input_file(arguments, WEIGHTS)
    ibtracs = input_file(arguments, IBTRACS)
    runs = forecast_run_directory(
        scenario, init, dataset, weights, load_dataset(ibtracs), arguments, out
    )
    return f"{', '.join(runs)} forecasts, atlases and {METADATA_FILE} written to {out}"
