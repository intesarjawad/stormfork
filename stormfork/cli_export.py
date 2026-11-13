"""Turn a run directory into the published evidence files.

A run directory holds metadata.json (the experiment metadata), one directory
per run (baseline, fork, control) with its atlases and the tracker's tracks.csv,
and the preview directory; a sweep also holds stops.json and a directory per
stop under stops/. stormfork run writes all but the tracks, and stormfork
rerun writes those too. Evidence is derived from tracks.csv alone,
except that a hindcast also reads its observed track from an IBTrACS CSV.
"""

import shutil
from pathlib import Path
from typing import get_args

import pandas as pd
from pydantic import TypeAdapter

from stormfork.cli_verify import ADJUSTMENT_RATIO_GATE
from stormfork.errors import UserError
from stormfork.export.archive import write_release_zip
from stormfork.export.divergence import Divergence, RunName, compute_divergence
from stormfork.export.manifest import (
    ExperimentMetadata,
    Manifest,
    SweepExport,
    export_experiment,
    manifest_paths,
)
from stormfork.export.metadata import SweepStop
from stormfork.export.observed import observed_track
from stormfork.export.reproduction import null_fork_max_track_diff_km
from stormfork.export.sweep import STOP_ATLAS, StopRun, sweep_evidence
from stormfork.scenarios import NULL_EDIT_VALUES
from stormfork.track.match import TRACK_COLUMNS

METADATA_FILE = "metadata.json"
STOPS_FILE = "stops.json"
TRACKS_FILE = "tracks.csv"
ATLASES_DIR = Path("dist/atlases")


def read_tracks(path: Path) -> pd.DataFrame:
    """Read the tracker's table for one run."""
    frame = pd.read_csv(path, dtype={"track_id": str}, parse_dates=["valid_time"])
    return frame.loc[:, list(TRACK_COLUMNS)]


def published_stops(run_directory: Path) -> tuple[str, list[StopRun]]:
    """Read a sweep's swept parameter and the stops it publishes, with their tracks and atlas.

    A stop is published unless the edit check refused it or its adjustment
    ratio is above the publication gate.

    Raises:
        UserError: The run directory lacks stops.json or a published stop's tracks.
    """
    path = run_directory / STOPS_FILE
    if not path.is_file():
        msg = f'"{run_directory}" has no {STOPS_FILE}'
        raise UserError(msg, f"write it with stormfork run <id> --out {run_directory}", 2)
    stops = TypeAdapter(list[SweepStop]).validate_json(path.read_bytes())
    runs: list[StopRun] = []
    for stop in stops:
        ratio = stop.adjustment_ratio
        if stop.refused is not None or (ratio is not None and ratio > ADJUSTMENT_RATIO_GATE):
            continue
        tracks = run_directory / stop.run / TRACKS_FILE
        if not tracks.is_file():
            msg = f"{tracks} does not exist"
            next_step = f"track {run_directory / stop.run}/forecast.nc with stormfork track"
            raise UserError(msg, next_step, 2)
        atlas = (run_directory / stop.run / STOP_ATLAS).read_bytes()
        runs.append(StopRun(stop.run, stop.value, read_tracks(tracks), atlas))
    return stops[0].param, runs


def sweep_divergence(
    run_directory: Path, tracks: dict[RunName, pd.DataFrame], divergence: Divergence
) -> tuple[Divergence, SweepExport | None]:
    """Add a sweep's stops to its null fork's divergence, and build its slider block.

    Raises:
        UserError: The run directory lacks stops.json or a published stop's tracks.
    """
    param, stops = published_stops(run_directory)
    return sweep_evidence(param, NULL_EDIT_VALUES[param], tracks, divergence, stops)


def export_release(run_directory: Path, out: Path, ibtracs: Path | None = None) -> str:
    """Export a run directory, then pack the experiment as dist/atlases/<id>.zip.

    The zip holds exactly the manifest and the files it names, so the same
    export gives the same bytes, and the release lock can pin it by SHA-256.
    """
    exported = export_run_directory(run_directory, out, ibtracs)
    manifest = Manifest.model_validate_json((out / "manifest.json").read_bytes())
    archive = ATLASES_DIR / f"{manifest.id}.zip"
    write_release_zip(out, manifest_paths(manifest), archive)
    return f"{exported}\nrelease zip written to {archive}"


def export_run_directory(run_directory: Path, out: Path, ibtracs: Path | None = None) -> str:
    """Write the manifest, tracks and divergence for a run directory into out.

    Atlases and previews are copied when out differs from the run directory.
    A hindcast needs the IBTrACS CSV for its observed track; other kinds refuse it.
    A sweep's run directory holds its zero-edit stop, so its manifest records how
    far that fork lies from the baseline; its divergence carries every stop's
    paired statistics, and a displacement sweep's manifest lists its stops.

    Raises:
        UserError: The run directory lacks its metadata, stops or tracker output,
            or the IBTrACS file is given for the wrong kind or missing for a hindcast.
        ValueError: A sweep lacks its fork tracks or they differ in coverage.
    """
    metadata_path = run_directory / METADATA_FILE
    if not metadata_path.is_file():
        msg = f'"{run_directory}" has no {METADATA_FILE}'
        raise UserError(msg, f"write it with stormfork run <id> --out {run_directory}", 2)
    metadata = ExperimentMetadata.model_validate_json(metadata_path.read_bytes())
    tracks: dict[RunName, pd.DataFrame] = {}
    for run in get_args(RunName):
        path = run_directory / run / TRACKS_FILE
        if path.is_file():
            tracks[run] = read_tracks(path)
    if (metadata.kind == "hindcast") != (ibtracs is not None):
        msg = "--ibtracs is for hindcasts, and a hindcast needs it"
        raise UserError(msg, "pass the IBTrACS CSV for a hindcast and omit it otherwise", 2)
    if "baseline" not in tracks:
        msg = f"{run_directory / 'baseline' / TRACKS_FILE} does not exist"
        raise UserError(msg, "run stormfork track on the run directory first", 2)
    if out.resolve() != run_directory.resolve():
        for name in (*tracks, "preview"):
            if (run_directory / name).is_dir():
                shutil.copytree(run_directory / name, out / name, dirs_exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    observed = (
        observed_track(ibtracs, metadata.init.storm.ibtracs_sid, metadata.init.time, metadata.steps)
        if ibtracs is not None
        else None
    )
    divergence, sweep, null_fork = compute_divergence(tracks), None, None
    if metadata.kind == "sweep":
        null_fork = null_fork_max_track_diff_km(tracks)
        divergence, sweep = sweep_divergence(run_directory, tracks, divergence)
    export_experiment(
        out,
        metadata,
        tracks,
        divergence,
        null_fork_max_track_diff_km=null_fork,
        observed=observed,
        sweep=sweep,
    )
    return f"manifest written to {out / 'manifest.json'}"
