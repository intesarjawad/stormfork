"""Edit an experiment's model input once per fork, and measure each fork's first step.

Nothing here needs the model stack: the edits, the edit check and the
adjustment ratio are pure numpy and scipy on xarray datasets. A sweep runs
one fork per value, its null value as the fork and the others as stops.
"""

from typing import NamedTuple

import xarray as xr
from pydantic import TypeAdapter

from stormfork.cli_perturb import moved_centre
from stormfork.errors import UserError
from stormfork.export.manifest import PerturbationInfo
from stormfork.export.metadata import SweepStop
from stormfork.perturb import (
    PhysicalBoundsError,
    Taper,
    adjustment_ratio,
    noise,
    relocate,
    scale_intensity,
    sst_anomaly,
    validate,
)
from stormfork.perturb.base import MIN_RADIUS_KM, taper_radius_km
from stormfork.scenarios import NULL_EDIT_VALUES, Scenario
from stormfork.track import StormStart

MSLP = "mean_sea_level_pressure"
STOPS_DIR = "stops"


class EditedInput(NamedTuple):
    """A fork's model input after its edit, and the land fraction of a relocated core."""

    inputs: xr.Dataset
    core_land_fraction: float | None


def parameter(edit: PerturbationInfo, name: str) -> float:
    """Read one numeric edit parameter, in the unit the recipe gives it.

    Raises:
        UserError: The recipe leaves the parameter out or gives it no number.
    """
    value = edit.params.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{edit.type} needs a number for {name}"
        raise UserError(msg, f"set edit.params.{name} in the recipe", 2)
    return float(value)


def moved_taper(scenario: Scenario) -> Taper | None:
    """Where a relocation puts the storm's taper; None for every other edit."""
    edit, placement = scenario.edit, scenario.taper
    if edit is None or placement is None or edit.type != "relocate":
        return None
    taper = Taper(*placement.centre, placement.radius_km)
    centre = moved_centre(taper, parameter(edit, "east_km"), parameter(edit, "north_km"))
    return Taper(*centre, taper.radius_km)


def apply_edit(scenario: Scenario, inputs: xr.Dataset) -> EditedInput:
    """Apply a fork's edit to the model inputs and check the result is physical.

    Raises:
        ValueError: The experiment has no edit.
        PhysicalBoundsError: The edited inputs leave the allowed ranges.
    """
    edit, placement = scenario.edit, scenario.taper
    if edit is None:
        msg = f"{scenario.id} has no edit to fork with"
        raise ValueError(msg)
    if placement is not None:
        taper = Taper(*placement.centre, placement.radius_km)
        if edit.type == "relocate":
            edited = relocate(
                inputs,
                taper,
                east_km=parameter(edit, "east_km"),
                north_km=parameter(edit, "north_km"),
            )
        else:
            edited = scale_intensity(inputs, taper, factor=parameter(edit, "factor"))
    elif edit.type == "sst_anomaly":
        centre = TypeAdapter(tuple[float, float]).validate_python(edit.params.get("centre"))
        edited = sst_anomaly(
            inputs,
            centre=centre,
            sigma_km=parameter(edit, "sigma_km"),
            delta_k=parameter(edit, "delta_k"),
        )
    else:
        edited = noise(
            inputs, amplitude=parameter(edit, "amplitude"), seed=int(parameter(edit, "seed"))
        )
    return EditedInput(edited, validate(inputs, edited, moved_taper=moved_taper(scenario)))


def fork_recipes(scenario: Scenario) -> dict[str, Scenario]:
    """Each fork an experiment rolls out, by run directory, as a single-edit recipe.

    A fork has one. A sweep's null value is its fork, beside the baseline, and
    every other value a stop under stops/<param>_<value>. A hindcast has none.
    """
    edit, sweep = scenario.edit, scenario.sweep
    if edit is None:
        return {}
    if sweep is None:
        return {"fork": scenario}
    forks: dict[str, Scenario] = {}
    for value in sweep.values:
        stop_edit = PerturbationInfo(type=edit.type, params={**edit.params, sweep.param: value})
        forks[sweep_run(sweep.param, value)] = scenario.model_copy(
            update={"kind": "fork", "sweep": None, "edit": stop_edit}
        )
    return forks


def same_forecast(recipe: Scenario, other: Scenario) -> bool:
    """Whether two recipes roll out the same forecast, whatever their id, title and limits."""
    text = {"id", "title", "limits"}
    return recipe.model_dump(exclude=text) == other.model_dump(exclude=text)


def sweep_run(param: str, value: float) -> str:
    """Name the run directory of one sweep value: fork for the null value, else a stop."""
    return "fork" if value == NULL_EDIT_VALUES[param] else f"{STOPS_DIR}/{param}_{value:g}"


def first_step_pressure(inputs: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    """Member 0's pressure at the init and after the first step, the two states the ratio reads."""
    initial = inputs[MSLP].isel(time=-1, drop=True).reset_coords(drop=True)
    first = forecast[MSLP].isel(member=0, time=0, drop=True).reset_coords(drop=True)
    return xr.concat([initial, first], dim="time").to_dataset()


def ratio_tapers(scenario: Scenario, storm: StormStart) -> list[Taper]:
    """Where a fork's adjustment ratio is measured, so every fork reports the same quantity.

    A tapered edit is measured in its own taper; a relocation also in the taper
    it moved to, since the edit disturbs both. An edit without a taper is
    measured in the taper a tapered edit of the storm would get at the init:
    the taper rule's radius for its gale-force radii, or the rule's smallest
    radius when none is recorded. Before the storm's first record it has no
    position and is not measured.
    """
    placement = scenario.taper
    if placement is None:
        if storm.position is None:
            return []
        radius_km = taper_radius_km(storm.gale_radii_nm) or MIN_RADIUS_KM
        return [Taper(*storm.position, radius_km)]
    tapers = [Taper(*placement.centre, placement.radius_km)]
    moved = moved_taper(scenario)
    return tapers if moved is None else [*tapers, moved]


def first_step_adjustment_ratio(
    scenario: Scenario, storm: StormStart, fork: xr.Dataset, baseline: xr.Dataset
) -> float | None:
    """Compare the fork's first-step pressure change with the baseline's, taking the largest.

    fork and baseline are first_step_pressure's two states for each run.

    Raises:
        DiagnosticError: The baseline's pressure does not change in an annulus.
    """
    tapers = ratio_tapers(scenario, storm)
    return max((adjustment_ratio(fork, baseline, taper) for taper in tapers), default=None)


def edit_forks(
    scenario: Scenario, inputs: xr.Dataset
) -> tuple[dict[str, EditedInput], dict[str, str]]:
    """Edit the input once per fork, setting aside the sweep stops the edit check refuses.

    Returns the edited inputs and the refusal messages, both by run directory.

    Raises:
        PhysicalBoundsError: A fork's edited input, or a sweep's null fork's,
            leaves the allowed ranges.
    """
    edited: dict[str, EditedInput] = {}
    refused: dict[str, str] = {}
    for run, recipe in fork_recipes(scenario).items():
        try:
            edited[run] = apply_edit(recipe, inputs)
        except PhysicalBoundsError as error:
            if run == "fork":
                raise
            refused[run] = str(error)
    return edited, refused


def sweep_stops(
    scenario: Scenario,
    storm: StormStart,
    edited: dict[str, EditedInput],
    refused: dict[str, str],
    pressure: dict[str, xr.Dataset],
) -> list[SweepStop]:
    """Describe each sweep value but the null fork, with its diagnostics or why it was refused."""
    if scenario.sweep is None:
        return []
    param = scenario.sweep.param
    recipes = fork_recipes(scenario)
    stops: list[SweepStop] = []
    for value in scenario.sweep.values:
        run = sweep_run(param, value)
        if run in refused:
            stops.append(SweepStop(run=run, param=param, value=value, refused=refused[run]))
        elif run != "fork":
            ratio = first_step_adjustment_ratio(
                recipes[run], storm, pressure[run], pressure["baseline"]
            )
            stops.append(
                SweepStop(
                    run=run,
                    param=param,
                    value=value,
                    adjustment_ratio=ratio,
                    core_land_fraction=edited[run].core_land_fraction,
                )
            )
    return stops
