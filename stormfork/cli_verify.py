"""Verify a published experiment from its manifest, tracks and divergence.

Hashes alone show the files are the published bytes; recomputing the divergence
from the tracks shows the published numbers follow from them. Pressure RMSE
needs the gridded fields and the Holm adjustment needs the sibling sweep stops,
so those two values are not recomputed.
"""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

import pandas as pd
from pydantic import JsonValue, TypeAdapter

from stormfork.cli_remote import Hero, Lock, download_experiment, fetch_lock
from stormfork.errors import UserError
from stormfork.export.divergence import Divergence, DivergenceStep, RunName, compute_divergence
from stormfork.export.manifest import GithubLogin, Manifest, TrackPoint, verify_manifest
from stormfork.export.reproduction import (
    NULL_FORK_TOLERANCE_KM,
    Reproduction,
    max_track_difference_km,
    null_fork_max_track_diff_km,
    tolerance_km,
)
from stormfork.track.match import TRACK_COLUMNS

PUBLISHED_DIR = Path("site/public/experiments")
HERO_FILE = Path("site/hero.json")
TOLERANCE = 1e-9
# Above this first-step pressure tendency ratio a fork is not published (shock at the taper edge).
ADJUSTMENT_RATIO_GATE = 1.5
EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
NOT_RECOMPUTED = {"stops": True, "steps": {"__all__": {"mslp_rmse_pa", "paired_p_holm"}}}


def tracks_from_json(data: bytes) -> dict[RunName, pd.DataFrame]:
    """Rebuild the per-run tracker tables that an export was written from."""
    points = TypeAdapter(list[TrackPoint]).validate_json(data)
    frame = pd.DataFrame([point.model_dump() for point in points])
    frame = frame.assign(valid_time=pd.to_datetime(frame["valid_time"]))
    tracks: dict[RunName, pd.DataFrame] = {}
    for run in get_args(RunName):
        rows = frame.loc[frame.run == run, list(TRACK_COLUMNS)]
        if len(rows):
            tracks[run] = pd.DataFrame(rows).reset_index(drop=True)
    return tracks


def first_difference(stored: JsonValue, fresh: JsonValue, where: str) -> str | None:
    """Name the first place two JSON documents differ by more than the tolerance."""
    if isinstance(stored, dict) and isinstance(fresh, dict):
        if stored.keys() != fresh.keys():
            return where
        pairs = ((f"{where}.{key}", stored[key], fresh[key]) for key in stored)
    elif isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return where
        pairs = (
            (f"{where}[{index}]", *pair)
            for index, pair in enumerate(zip(stored, fresh, strict=True))
        )
    elif isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        return None if abs(stored - fresh) <= TOLERANCE else where
    else:
        return None if stored == fresh else where
    return next((found for path, a, b in pairs if (found := first_difference(a, b, path))), None)


def has_observed(directory: Path, manifest: Manifest) -> bool:
    """Whether the published tracks include the observed run."""
    data = (directory / manifest.tracks["json"].path).read_bytes()
    return any(
        point.run == "observed" for point in TypeAdapter(list[TrackPoint]).validate_json(data)
    )


def check_divergence(directory: Path, manifest: Manifest) -> Divergence:
    """Recompute divergence.json from tracks.json with the export's own functions.

    Raises:
        ValueError: A recomputed value differs from the published one by more than 1e-9.
    """
    stored = Divergence.model_validate_json((directory / manifest.divergence.path).read_bytes())
    tracks = tracks_from_json((directory / manifest.tracks["json"].path).read_bytes())
    fresh = compute_divergence(tracks, grid_resolution_deg=stored.grid_resolution_deg)
    published, recomputed = (
        json.loads(item.model_dump_json(exclude=NOT_RECOMPUTED)) for item in (stored, fresh)
    )
    difference = first_difference(published, recomputed, "divergence")
    if difference is not None:
        tracks_path = manifest.tracks["json"].path
        msg = f"{manifest.divergence.path} differs from {tracks_path} at {difference}"
        raise ValueError(msg)
    return stored


def check_null_fork(directory: Path, manifest: Manifest) -> float:
    """Check the recorded baseline-to-fork distance of a sweep's zero-edit run.

    Returns:
        The recorded distance in km.

    Raises:
        ValueError: The distance is not recorded, does not follow from the
            tracks, or exceeds the null-fork tolerance.
    """
    recorded = manifest.diagnostics.null_fork_max_track_diff_km
    if recorded is None:
        msg = "a sweep manifest must record diagnostics.null_fork_max_track_diff_km"
        raise ValueError(msg)
    tracks = tracks_from_json((directory / manifest.tracks["json"].path).read_bytes())
    fresh = null_fork_max_track_diff_km(tracks)
    if abs(recorded - fresh) > TOLERANCE:
        msg = f"null_fork_max_track_diff_km is {recorded} but the tracks give {fresh}"
        raise ValueError(msg)
    if recorded > NULL_FORK_TOLERANCE_KM:
        msg = (
            f"the zero-edit fork is {recorded:.1f} km from the baseline, "
            f"over the {NULL_FORK_TOLERANCE_KM:.1f} km limit"
        )
        raise ValueError(msg)
    return recorded


def check_adjustment_ratio(manifest: Manifest) -> None:
    """Hold a recorded adjustment ratio to the publication gate.

    Raises:
        UserError: The ratio is above the gate.
    """
    ratio = manifest.diagnostics.adjustment_ratio
    if ratio is not None and ratio > ADJUSTMENT_RATIO_GATE:
        msg = f"adjustment ratio {ratio} is above the {ADJUSTMENT_RATIO_GATE} publication gate"
        next_step = "leave the fork unpublished and record why in docs/demo-gate.md"
        raise UserError(msg, next_step, 1)


def check_bounds(manifest: Manifest) -> None:
    """Require an edited experiment to record that its edited input passed the edit check.

    Raises:
        UserError: A fork or sweep does not record a passed bounds check.
    """
    if manifest.kind != "hindcast" and manifest.diagnostics.bounds_check != "passed":
        msg = f"{manifest.id} records no passed bounds check"
        raise UserError(msg, f"rerun it with stormfork run {manifest.id}, which checks the edit", 1)


def separation_lines(row: DivergenceStep) -> list[str]:
    """Report one step's fork and control separation from the baseline and its paired test."""

    def show(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    return [
        f"step: {row.step}",
        f"fork_sep_km: {show(row.fork_sep_km, 1)}",
        f"control_sep_km: {show(row.control_sep_km, 1)}",
        f"paired_p: {show(row.paired_p, 4)}",
    ]


def verify_directory(
    directory: Path, hero: Hero | None = None, *, expected_id: str | None = None
) -> str:
    """Verify hashes and divergence, then report separation at the landfall step.

    A sweep's manifest must also record a zero-edit fork within the null-fork
    tolerance of its baseline, and the report gains a line for that distance.

    The landfall step comes from the hero record when it names this experiment;
    otherwise the last step is reported.

    Raises:
        UserError: A fork or sweep lacks its passed bounds check, or its
            adjustment ratio is above the publication gate.
        ValueError: A hash or recomputed value differs, the manifest names another
            experiment than expected_id, a hindcast lacks its observed track,
            a sweep's null-fork check fails, or the landfall step is absent.
    """
    manifest = verify_manifest(directory)
    if expected_id is not None and manifest.id != expected_id:
        msg = f'the release holds "{manifest.id}" under the id "{expected_id}"'
        raise ValueError(msg)
    check_bounds(manifest)
    check_adjustment_ratio(manifest)
    divergence = check_divergence(directory, manifest)
    if manifest.kind == "hindcast" and not has_observed(directory, manifest):
        msg = f"{manifest.tracks['json'].path} has no observed track"
        raise ValueError(msg)
    null_fork = check_null_fork(directory, manifest) if manifest.kind == "sweep" else None
    if not divergence.steps:
        msg = f"{manifest.divergence.path} has no steps"
        raise ValueError(msg)
    wanted = hero.landfall_step if hero and hero.id == manifest.id else None
    step = wanted if wanted is not None else max(row.step for row in divergence.steps)
    row = next((row for row in divergence.steps if row.step == step), None)
    if row is None:
        msg = f"{manifest.divergence.path} has no step {step}"
        raise ValueError(msg)

    lines = [manifest.id, *separation_lines(row)]
    if null_fork is not None:
        lines.append(f"null_fork_max_track_diff_km: {null_fork:.3f}")
    return "\n".join((*lines, "PASS"))


def read_hero() -> Hero | None:
    """Read the hero record when running inside a clone."""
    if not HERO_FILE.is_file():
        return None
    return Hero.model_validate_json(HERO_FILE.read_bytes())


def reproduce_directory(
    published: Path, rerun: Path, by: str, *, expected_id: str | None = None
) -> str:
    """Compare a rerun's exported tracks with the published ones and print one entry.

    Both directories are hash-checked first. The entry passes when no member or
    ensemble-mean position differs by more than the runtime tolerance.

    Raises:
        UserError: The rerun directory has not been exported.
        ValueError: A hash differs, the rerun is another experiment or the tracks
            do not cover the same samples.
    """
    if not (rerun / "manifest.json").is_file():
        msg = f'"{rerun}" has no manifest.json'
        raise UserError(msg, "run stormfork export on the run directory first", 2)
    expected = verify_manifest(published)
    if expected_id is not None and expected.id != expected_id:
        msg = f'the release holds "{expected.id}" under the id "{expected_id}"'
        raise ValueError(msg)
    actual = verify_manifest(rerun)
    if actual.id != expected.id:
        msg = f'the rerun holds "{actual.id}", not "{expected.id}"'
        raise ValueError(msg)
    difference = max_track_difference_km(
        tracks_from_json((published / expected.tracks["json"].path).read_bytes()),
        tracks_from_json((rerun / actual.tracks["json"].path).read_bytes()),
    )
    entry = Reproduction(
        by=by,
        runtime=f"{actual.runtime.kind}: {actual.runtime.device}",
        date=actual.created.date(),
        code_version=actual.model.code_version,
        max_track_diff_km=difference,
        passed=difference <= tolerance_km(expected.runtime.kind, actual.runtime.kind),
    )
    return entry.model_dump_json()


def check_checkout_copy(directory: Path, experiment: str) -> None:
    """Require every file a checkout's published manifest lists before hashing them.

    Git holds each published manifest, its tracks and previews, but the atlases
    come from the release, so a fresh clone lacks them until they are fetched.

    Raises:
        UserError: A listed file is absent.
    """
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
    listed = [artifact.path for artifact in (*manifest.tracks.values(), manifest.divergence)]
    listed.extend(path for assets in manifest.runs.values() for path in assets)
    absent = [path for path in listed if not (directory / path).is_file()]
    if absent:
        msg = f"{experiment} lacks {len(absent)} listed files, first {absent[0]}"
        raise UserError(msg, "fetch the atlases with uv run python scripts/fetch_atlases.py", 2)


def check_reproduction_arguments(reproduction: Path | None, by: str | None) -> None:
    """Require the run directory and the reproducer together, and a GitHub login.

    Raises:
        UserError: One is given without the other, or the login is malformed.
    """
    if (reproduction is None) != (by is None):
        msg = "--reproduction and --by must be supplied together"
        raise UserError(msg, "supply both or neither", 2)
    if by is not None:
        try:
            TypeAdapter(GithubLogin).validate_python(by)
        except ValueError as error:
            msg = f'"{by}" is not a GitHub user name'
            raise UserError(msg, "pass your GitHub login to --by", 2) from error


def verify_target(
    target: str | None, *, reproduction: Path | None = None, by: str | None = None
) -> str:
    """Verify an existing path, a published experiment id or the hero.

    An id without a local copy is downloaded from the pinned release into a
    temporary directory. With a reproduction run directory, the published
    tracks are compared with the rerun's instead of being verified.
    """
    check_reproduction_arguments(reproduction, by)
    hero = read_hero()
    lock: Lock | None = None
    if target is None:
        if hero is None:
            lock = fetch_lock()
            hero = lock.hero
        target = hero.id

    def check(directory: Path, record: Hero | None, expected_id: str | None = None) -> str:
        if reproduction is not None and by is not None:
            return reproduce_directory(directory, reproduction, by, expected_id=expected_id)
        return verify_directory(directory, record, expected_id=expected_id)

    if Path(target).exists() or not EXPERIMENT_ID.fullmatch(target):
        path = Path(target)
        return check(path.parent if path.name == "manifest.json" else path, hero)
    published = PUBLISHED_DIR / target
    if published.is_dir():
        check_checkout_copy(published, target)
        return check(published, hero)
    lock = lock or fetch_lock()
    with TemporaryDirectory() as unpacked:
        directory = download_experiment(lock, target, Path(unpacked))
        return check(directory, hero or lock.hero, expected_id=target)
