"""Read the published experiment list from its YAML files.

Each file in the experiments directory defines one experiment: its storm,
initialization times, initial-condition source, edit and seeds, with an optional
title and the limits shown beside its results. Times are UTC.
Positions are latitude and longitude in degrees, distances are km and
temperatures are K, matching the edits in ``stormfork.perturb``.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import Field, field_validator, model_validator

from stormfork.errors import UserError
from stormfork.export.divergence import EvidenceModel
from stormfork.export.manifest import InitSource, PerturbationInfo, Seeds, Text

EXPERIMENTS_DIR = Path("experiments")
Kind = Literal["fork", "sweep", "hindcast"]
Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=360)]
EDIT_PARAMETERS = {
    "relocate": frozenset({"east_km", "north_km"}),
    "scale_intensity": frozenset({"factor"}),
    "sst_anomaly": frozenset({"centre", "sigma_km", "delta_k"}),
    "noise": frozenset({"amplitude", "seed"}),
}
TAPERED_EDITS = frozenset({"relocate", "scale_intensity"})
# The value of each sweepable parameter that leaves the input as it was.
NULL_EDIT_VALUES = {
    "east_km": 0.0,
    "north_km": 0.0,
    "factor": 1.0,
    "delta_k": 0.0,
    "amplitude": 0.0,
}
CYCLE_HOURS = 6


class TaperPlacement(EvidenceModel):
    """Where an edit's taper sits at the experiment's init time."""

    centre: tuple[Latitude, Longitude]
    radius_km: Annotated[float, Field(gt=0)]


class Sweep(EvidenceModel):
    """One edit parameter taken at several values, one run per value."""

    param: Text
    values: Annotated[list[float], Field(min_length=2)]


class Scenario(EvidenceModel):
    """One experiment; a hindcast has several inits and no edit.

    The title and limits are the text the manifest carries. A published experiment
    needs both; a recipe someone runs for themselves may leave them out.
    """

    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]*$")]
    title: Text | None = None
    kind: Kind
    storm: Text
    init: Annotated[tuple[datetime, ...], Field(min_length=1)]
    source: InitSource
    edit: PerturbationInfo | None = None
    sweep: Sweep | None = None
    taper: TaperPlacement | None = None
    seeds: Seeds
    limits: list[Text] = Field(default_factory=list)

    @field_validator("init", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> object:
        return cast("list[object]", value) if isinstance(value, list) else [value]

    @field_validator("init")
    @classmethod
    def _utc_cycle_times(cls, value: tuple[datetime, ...]) -> tuple[datetime, ...]:
        """Normalise to naive UTC, as the rest of the package does.

        Raises:
            ValueError: An init is off the six-hour UTC cycle or repeated.
        """
        times = tuple(t.astimezone(UTC).replace(tzinfo=None) if t.tzinfo else t for t in value)
        for time in times:
            if time.hour % CYCLE_HOURS or time.minute or time.second or time.microsecond:
                msg = f"init {time.isoformat()} is off the six-hour UTC cycle"
                raise ValueError(msg)
        if len(set(times)) != len(times):
            msg = "init times must be distinct"
            raise ValueError(msg)
        return times

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        """Tie edit, sweep, taper and init count to the experiment kind.

        Raises:
            ValueError: The fields do not describe a runnable experiment.
        """
        if self.kind == "hindcast":
            if self.edit or self.sweep or self.taper:
                msg = "a hindcast has no edit, sweep or taper"
                raise ValueError(msg)
            return self
        if self.edit is None or self.edit.type not in EDIT_PARAMETERS:
            msg = f"{self.kind} needs an edit of type {', '.join(EDIT_PARAMETERS)}"
            raise ValueError(msg)
        if len(self.init) != 1:
            msg = f"{self.kind} needs exactly one init"
            raise ValueError(msg)
        if (self.kind == "sweep") != (self.sweep is not None):
            msg = "only a sweep has a sweep block, and a sweep needs one"
            raise ValueError(msg)
        names: set[str] = set(self.edit.params)
        if self.sweep:
            names.add(self.sweep.param)
        unknown = names - EDIT_PARAMETERS[self.edit.type]
        if unknown or (self.sweep and self.sweep.param in self.edit.params):
            msg = f"{self.edit.type} parameters are invalid: {sorted(names)}"
            raise ValueError(msg)
        if (self.edit.type in TAPERED_EDITS) != (self.taper is not None):
            msg = "relocate and scale_intensity need a taper; other edits must not have one"
            raise ValueError(msg)
        if self.sweep:
            self._check_sweep_values(self.sweep)
        return self

    @staticmethod
    def _check_sweep_values(sweep: Sweep) -> None:
        """Require distinct values, one of them the null fork that leaves the input unedited.

        Raises:
            ValueError: A value repeats, or the null value is absent or undefined.
        """
        if len(set(sweep.values)) != len(sweep.values):
            msg = "sweep values must be distinct"
            raise ValueError(msg)
        if sweep.param not in NULL_EDIT_VALUES:
            msg = f"{sweep.param} has no null value, so a sweep over it has no null fork"
            raise ValueError(msg)
        null = NULL_EDIT_VALUES[sweep.param]
        if null not in sweep.values:
            msg = f"a sweep over {sweep.param} must include {null:g}, the null fork"
            raise ValueError(msg)


def load_scenarios(directory: Path = EXPERIMENTS_DIR) -> tuple[Scenario, ...]:
    """Read every ``*.yaml`` file, sorted by id; a file is named after its id.

    Raises:
        UserError: The directory is missing or holds no experiments.
        ValueError: A file is malformed or its name differs from its id.
    """
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        msg = f'no experiment files in "{directory}"'
        raise UserError(msg, "run from the repository root or pass --dir", 2)
    scenarios: list[Scenario] = []
    for path in paths:
        scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if scenario.id != path.stem:
            msg = f'{path}: id "{scenario.id}" must match the file name'
            raise ValueError(msg)
        scenarios.append(scenario)
    return tuple(scenarios)


def format_time(time: datetime) -> str:
    """Write a naive UTC time as it appears in the experiment files."""
    return time.strftime("%Y-%m-%dT%H:%MZ")


def list_lines(scenarios: tuple[Scenario, ...]) -> str:
    """Print one line per init: id, kind, source and init time."""
    rows = [(s.id, s.kind, s.source, format_time(time)) for s in scenarios for time in s.init]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths, strict=True))
        + f"  {row[3]}"
        for row in rows
    )
