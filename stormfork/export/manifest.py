import csv
import hashlib
import io
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, NamedTuple, Self

import pandas as pd
from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from stormfork.export.divergence import Divergence, EvidenceModel, Nonnegative, Probability, RunName
from stormfork.export.quantise import Encoding, field_for_file, field_metadata
from stormfork.track.match import TRACK_COLUMNS

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
GithubLogin = Annotated[str, Field(max_length=39, pattern=r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Index = Annotated[int, Field(ge=0)]
ExperimentId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
# Version 2 lets the storm's position at init be null before its first record.
SCHEMA_VERSION: Final = 2
TrackFormat = Literal["json", "geojson", "csv"]
InitSource = Literal["era5", "hres_sample", "ifs_open_hybrid"]
TrackRun = Literal["baseline", "fork", "control", "observed"]
HINDCAST_LIMIT = (
    "Initial conditions are reconstructed from ERA5 reanalysis, not the analysis available "
    "at the time, so a hindcast shows what the model does from a reconstructed state and is "
    "not the forecast it would have issued then."
)


class ModelInfo(EvidenceModel):
    """Checkpoint identity and the model implementation used for a rollout."""

    name: Text
    checkpoint: Text
    md5: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    code_version: Text


class RuntimeInfo(EvidenceModel):
    """Compute backend and concrete device description."""

    kind: Literal["cpu", "gpu", "tpu"]
    device: Text


class StormInfo(EvidenceModel):
    """Target storm at initialization, with positions in degrees.

    An init before the storm's first six-hourly IBTrACS record has no position,
    so lat and lon are both null there and nowhere else; first_record_time says
    when the storm appears.
    """

    ibtracs_sid: Text
    lat: Annotated[float, Field(ge=-90, le=90)] | None
    lon: Annotated[float, Field(ge=-180, le=360)] | None
    first_record_time: datetime


class InitialState(EvidenceModel):
    """Input provenance sufficient to locate the initial target storm."""

    source: InitSource
    time: datetime
    storm: StormInfo

    @model_validator(mode="after")
    def check_storm_position(self) -> Self:
        """Allow a missing position only for an init before the storm's first record.

        Raises:
            ValueError: The position is half missing, missing after the first
                record, or present before it.
        """
        storm = self.storm
        if (storm.lat is None) != (storm.lon is None):
            msg = "the storm's lat and lon must be both present or both null"
            raise ValueError(msg)
        if (storm.lat is None) != (self.time < storm.first_record_time):
            msg = "the storm's position is null exactly when the init precedes its first record"
            raise ValueError(msg)
        return self


class PerturbationInfo(EvidenceModel):
    """Edit type and its JSON-serializable physical parameters."""

    type: Text
    params: dict[str, JsonValue]


class Seeds(EvidenceModel):
    """The fork shares baseline seeds; control uses baseline plus 1000."""

    baseline: Index
    control: Index

    @model_validator(mode="after")
    def check_control(self) -> Self:
        """Keep the independent control reproducible from the baseline seed.

        Raises:
            ValueError: The control seed does not use the fixed offset.
        """
        if self.control != self.baseline + 1000:
            msg = "control seed must equal baseline seed + 1000"
            raise ValueError(msg)
        return self


class ExportField(EvidenceModel):
    """The exact decoding contract for one of the five exported fields."""

    name: str
    files: list[str]
    range: tuple[float, float]
    encoding: Encoding
    resolution_deg: Literal[1, 2]

    @model_validator(mode="after")
    def check_contract(self) -> Self:
        """Validate decoding against the quantiser's fixed ranges.

        Raises:
            ValueError: Metadata does not describe a supported field.
        """
        if not self.files:
            msg = "field files must not be empty"
            raise ValueError(msg)
        field = field_for_file(self.files[0])
        if (
            self.name != field.name
            or tuple(self.files) != field.files
            or self.range != (field.lower, field.upper)
            or self.encoding != field.encoding
            or (self.name != "wind10" and self.resolution_deg != 1)
        ):
            msg = "field metadata must match the quantisation contract"
            raise ValueError(msg)
        return self


class Diagnostics(EvidenceModel):
    """Reported balance, land overlap, 24-hour intensity drift in hPa and null-fork distance.

    core_land_fraction is the area-weighted land fraction inside a relocated
    core. bounds_check is "passed" once the edited input has passed
    stormfork.perturb.validate; an edit that fails is never rolled out, and a
    hindcast has no edit to check. null_fork_max_track_diff_km is the largest
    member or ensemble-mean distance in km between the baseline and a sweep's
    zero-edit fork.
    """

    adjustment_ratio: Nonnegative | None = None
    core_land_fraction: Probability | None = None
    bounds_check: Literal["passed"] | None = None
    intensity_drift_24h_hpa: float | None = None
    null_fork_max_track_diff_km: Nonnegative | None = None


def hindcast_init_id(parent_id: str, init: datetime) -> str:
    """Name one init of a hindcast: the hindcast's id and the init as YYYYMMDDHH."""
    return f"{parent_id}-{init:%Y%m%d%H}"


class ExperimentMetadata(EvidenceModel):
    """Caller-supplied provenance, with no clock or filesystem dependencies.

    run_by is the GitHub login of a contributor who ran the experiment; pull
    request checks hold it to the author who adds the experiment. Each init of
    a hindcast is its own experiment, whose parent_id is the hindcast's id.
    """

    schema_version: Literal[2]
    id: ExperimentId
    parent_id: ExperimentId | None = None
    title: Text
    kind: Literal["fork", "sweep", "hindcast"]
    model: ModelInfo
    runtime: RuntimeInfo
    init: InitialState
    perturbation: PerturbationInfo
    seeds: Seeds
    members: Annotated[int, Field(gt=0)]
    steps: Annotated[int, Field(ge=0)]
    fields: list[ExportField]
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    limits: Annotated[list[Text], Field(min_length=1)]
    attribution: Annotated[list[Text], Field(min_length=1)]
    created: datetime
    run_by: GithubLogin | None = None

    @model_validator(mode="after")
    def check_hindcast_limits(self) -> Self:
        """Require a hindcast to state how its initial conditions were reconstructed.

        Raises:
            ValueError: A hindcast's limits lack the reconstruction sentence.
        """
        if self.kind == "hindcast" and HINDCAST_LIMIT not in self.limits:
            msg = f"a hindcast's limits must include: {HINDCAST_LIMIT}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_parent(self) -> Self:
        """Tie a hindcast init's id to its parent hindcast and its init time.

        Raises:
            ValueError: A hindcast lacks its parent or its id does not name its
                init, or another kind has a parent.
        """
        if self.kind != "hindcast":
            if self.parent_id is not None:
                msg = "only a hindcast init has a parent_id"
                raise ValueError(msg)
            return self
        if self.parent_id is None or self.id != hindcast_init_id(self.parent_id, self.init.time):
            msg = "a hindcast init's id must be its parent_id and init as <parent_id>-YYYYMMDDHH"
            raise ValueError(msg)
        return self

    @field_validator("fields")
    @classmethod
    def check_fields(cls, fields: list[ExportField]) -> list[ExportField]:
        """Require all five fields exactly once.

        Raises:
            ValueError: Fields are missing, duplicated or unknown.
        """
        if sorted(field.name for field in fields) != sorted(
            item["name"] for item in field_metadata()
        ):
            msg = "fields must contain each of the five exported fields exactly once"
            raise ValueError(msg)
        return fields


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        msg = f"artifact path must be a relative POSIX path: {value!r}"
        raise ValueError(msg)
    return value


class Artifact(EvidenceModel):
    """Relative export path and SHA-256 of the exact published bytes."""

    path: str
    sha256: Sha256

    _check_path = field_validator("path")(_relative_path)


class PublishedStop(EvidenceModel):
    """One sweep value on the displacement slider, with the files the site loads for it.

    Paths are relative to the experiment directory. sha256 hashes the files
    only this stop publishes; the null stop reads the experiment's own tracks,
    divergence and fork atlas, which the rest of the manifest hashes.
    """

    id: Text
    direction: Literal["east", "west", "north", "south", "none"]
    distance_km: Nonnegative
    tracks: str
    atlas: str
    divergence: str
    sha256: dict[str, Sha256]

    _check_paths = field_validator("tracks", "atlas", "divergence")(_relative_path)

    @property
    def paths(self) -> set[str]:
        """The files the site loads for this stop."""
        return {self.tracks, self.atlas, self.divergence}


class SweepInfo(EvidenceModel):
    """The published stops of a displacement sweep, in slider order.

    name is the swept parameter and baseline the id of the null stop.
    """

    name: Text
    baseline: Text
    stops: Annotated[list[PublishedStop], Field(min_length=1)]

    @model_validator(mode="after")
    def check_stops(self) -> Self:
        """Require unique stop ids, one of them the null stop.

        Raises:
            ValueError: A stop id repeats or the null stop is not a stop.
        """
        ids = [stop.id for stop in self.stops]
        if len(set(ids)) != len(ids) or self.baseline not in ids:
            msg = "sweep stops need unique ids, and the null stop must be one of them"
            raise ValueError(msg)
        return self


class SweepExport(NamedTuple):
    """A sweep's slider block and the stop files it hashes, by relative path."""

    info: SweepInfo
    files: Mapping[str, bytes]


class Manifest(ExperimentMetadata):
    """Validated provenance and hashes for the complete published comparison."""

    runs: dict[RunName, dict[str, Sha256]]
    tracks: dict[TrackFormat, Artifact]
    divergence: Artifact
    sweep: SweepInfo | None = None

    @model_validator(mode="after")
    def check_assets(self) -> Self:
        """Require evidence for each comparison, all decoding files and every sweep stop's files.

        Raises:
            ValueError: A run, track format or required PNG is absent, or a
                sweep stop names a file the manifest does not hash.
        """
        required = {"baseline"} if self.kind == "hindcast" else {"baseline", "fork", "control"}
        if not required.issubset(self.runs) or set(self.tracks) != {"json", "geojson", "csv"}:
            msg = "manifest requires comparison runs and json, geojson, csv tracks"
            raise ValueError(msg)
        for run, assets in self.runs.items():
            for path in assets:
                _relative_path(path)
            expected = {f"{run}/{file}.png" for field in self.fields for file in field.files}
            if not expected.issubset(assets):
                msg = f"{run} is missing required PNG hashes"
                raise ValueError(msg)
        if self.sweep is None:
            return self
        if self.kind != "sweep":
            msg = "only a sweep publishes sweep stops"
            raise ValueError(msg)
        hashed = _hashed_paths(self)
        for stop in self.sweep.stops:
            if not stop.paths <= hashed:
                msg = f"sweep stop {stop.id} names a file the manifest does not hash"
                raise ValueError(msg)
        return self


class TrackPoint(EvidenceModel):
    """One sample, with string track IDs and longitudes in [-180, 180).

    String IDs keep CSV and JSON identities identical without a CSV type schema.
    The observed run is the IBTrACS truth of a hindcast, as member 0.
    """

    run: TrackRun
    member: Index
    step: Index
    valid_time: datetime
    lat: Annotated[float, Field(ge=-90, le=90)]
    lon: float
    mslp_pa: Nonnegative
    wind_ms: Nonnegative
    track_id: str

    @field_validator("track_id", mode="before")
    @classmethod
    def track_id_text(cls, value: object) -> str:
        """Preserve numeric tracker identifiers as text across all formats.

        Raises:
            TypeError: The tracker identifier is neither a string nor an integer.
        """
        if not isinstance(value, (str, int)):
            msg = "track_id must be a string or integer"
            raise TypeError(msg)
        return str(value)

    @field_validator("lon")
    @classmethod
    def wrap_longitude(cls, longitude: float) -> float:
        """Use the same canonical longitude in every download format."""
        return (longitude + 180) % 360 - 180


def track_exports(
    tracks: Mapping[RunName, pd.DataFrame], observed: pd.DataFrame | None = None
) -> dict[str, bytes]:
    """Derive all formats from canonical tracks.json records, preserving samples.

    The observed track, when given, is written under the run name "observed".

    GeoJSON properties retain step, intensity and track identity to permit a
    lossless round trip. A singleton is a repeated-coordinate LineString with
    sample_count=1; the duplicate is geometry only, never an invented sample.

    Raises:
        ValueError: Track samples are invalid or repeat a member/track step.
        TypeError: A track identifier is neither a string nor an integer.
    """
    points: list[TrackPoint] = []
    adapter = TypeAdapter(list[TrackPoint])
    frames: list[tuple[str, pd.DataFrame]] = [*tracks.items()]
    if observed is not None:
        frames.append(("observed", observed))
    for run, frame in frames:
        samples = frame.loc[:, list(TRACK_COLUMNS)].copy()
        samples["run"] = run
        points.extend(
            adapter.validate_json(
                samples.to_json(
                    orient="records", date_format="iso", date_unit="us", double_precision=15
                )
            )
        )
    points.sort(key=lambda point: (point.run, point.member, str(point.track_id), point.step))
    keys = [(point.run, point.member, point.track_id, point.step) for point in points]
    if len(set(keys)) != len(keys):
        msg = "tracks must have one sample per run, member, track and step"
        raise ValueError(msg)
    records = [point.model_dump(mode="json") for point in points]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=[*TRACK_COLUMNS, "run"])
    writer.writeheader()
    writer.writerows(records)
    groups: dict[tuple[str, int, str], list[dict[str, JsonValue]]] = {}
    # Parsing the canonical JSON keeps the secondary formats tied to its schema.
    canonical = adapter.dump_json(points)
    for point in adapter.validate_json(canonical):
        groups.setdefault((point.run, point.member, point.track_id), []).append(
            point.model_dump(mode="json")
        )
    features: list[dict[str, object]] = []
    for (run, member, track_id), samples in groups.items():
        coordinates = [[point["lon"], point["lat"]] for point in samples]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates if len(samples) > 1 else coordinates * 2,
                },
                "properties": {
                    "run": run,
                    "member": member,
                    "track_id": track_id,
                    "sample_count": len(samples),
                    "valid_times": [point["valid_time"] for point in samples],
                    "steps": [point["step"] for point in samples],
                    "mslp_pa": [point["mslp_pa"] for point in samples],
                    "wind_ms": [point["wind_ms"] for point in samples],
                },
            }
        )
    return {
        "tracks.json": canonical + b"\n",
        "tracks.csv": csv_buffer.getvalue().encode(),
        "tracks.geojson": (
            json.dumps({"type": "FeatureCollection", "features": features}, allow_nan=False) + "\n"
        ).encode(),
    }


def _local_path(directory: Path, relative: str) -> Path:
    path = directory / _relative_path(relative)
    if not path.resolve().is_relative_to(directory.resolve()):
        msg = f"artifact escapes export directory: {relative}"
        raise ValueError(msg)
    return path


def export_experiment(  # noqa: PLR0913
    directory: Path,
    metadata: ExperimentMetadata,
    tracks: Mapping[RunName, pd.DataFrame],
    divergence: Divergence,
    *,
    adjustment_ratio: float | None = None,
    null_fork_max_track_diff_km: float | None = None,
    observed: pd.DataFrame | None = None,
    sweep: SweepExport | None = None,
) -> Manifest:
    """Write evidence beside existing run PNGs, hashing the exact file bytes.

    The caller supplies provenance, any adjustment ratio and any null-fork
    distance; no perturbation or model imports are needed. Missing diagnostics
    remain null, never fabricated.
    A hindcast carries its observed track and no other experiment does. A
    sweep's stop files are written with the rest, and its block names them.

    Raises:
        ValueError: Provenance, tracks or assets violate the export contract.
        OSError: A required PNG is absent or an export cannot be written.
    """
    if (metadata.kind == "hindcast") != (observed is not None):
        msg = "only a hindcast has an observed track, and a hindcast needs one"
        raise ValueError(msg)
    exports = track_exports(tracks, observed)
    exports["divergence.json"] = (divergence.model_dump_json(indent=2) + "\n").encode()
    diagnostics = metadata.diagnostics.model_dump()
    diagnostics["intensity_drift_24h_hpa"] = divergence.intensity_drift_24h_hpa
    if adjustment_ratio is not None:
        diagnostics["adjustment_ratio"] = adjustment_ratio
    if null_fork_max_track_diff_km is not None:
        diagnostics["null_fork_max_track_diff_km"] = null_fork_max_track_diff_km
    runs: dict[RunName, dict[str, str]] = {}
    for run in tracks:
        names = [file for field in metadata.fields for file in field.files]
        paths = [f"{run}/{file}.png" for file in names]
        paths.extend(
            f"preview/{run}_{file}_f0.png"
            for file in names
            if _local_path(directory, f"preview/{run}_{file}_f0.png").exists()
        )
        runs[run] = {
            path: hashlib.sha256(_local_path(directory, path).read_bytes()).hexdigest()
            for path in paths
        }
    artifacts = {
        path: Artifact(path=path, sha256=hashlib.sha256(data).hexdigest())
        for path, data in exports.items()
    }
    manifest = Manifest.model_validate(
        {
            **metadata.model_dump(),
            "diagnostics": diagnostics,
            "runs": runs,
            "tracks": {
                format_: artifacts[f"tracks.{format_}"] for format_ in ("json", "geojson", "csv")
            },
            "divergence": artifacts["divergence.json"],
            "sweep": sweep.info if sweep is not None else None,
        }
    )
    if sweep is not None:
        exports.update(sweep.files)
    for path, data in exports.items():
        _local_path(directory, path).parent.mkdir(parents=True, exist_ok=True)
        _local_path(directory, path).write_bytes(data)
    _local_path(directory, "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _hashed_artifacts(manifest: Manifest) -> list[Artifact]:
    artifacts = [*manifest.tracks.values(), manifest.divergence]
    hashes = [*manifest.runs.values()]
    if manifest.sweep is not None:
        hashes.extend(stop.sha256 for stop in manifest.sweep.stops)
    artifacts.extend(
        Artifact(path=path, sha256=digest) for assets in hashes for path, digest in assets.items()
    )
    return artifacts


def _hashed_paths(manifest: Manifest) -> set[str]:
    return {artifact.path for artifact in _hashed_artifacts(manifest)}


def manifest_paths(manifest: Manifest) -> set[str]:
    """Every file a manifest names, and the manifest itself, relative to its directory."""
    return {"manifest.json", *_hashed_paths(manifest)}


def verify_manifest(directory: Path) -> Manifest:
    """Validate a manifest and recompute every referenced artifact hash.

    Raises:
        ValueError: Schema validation fails or an artifact hash differs.
        OSError: The manifest or a referenced artifact cannot be read.
    """
    manifest = Manifest.model_validate_json(_local_path(directory, "manifest.json").read_bytes())
    for artifact in _hashed_artifacts(manifest):
        actual = hashlib.sha256(_local_path(directory, artifact.path).read_bytes()).hexdigest()
        if actual != artifact.sha256:
            msg = f"{artifact.path} SHA-256 {actual} does not match the manifest {artifact.sha256}"
            raise ValueError(msg)
    return manifest
