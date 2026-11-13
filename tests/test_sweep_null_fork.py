import json
import math
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from stormfork import cli_verify
from stormfork.cli_export import export_run_directory
from stormfork.export.divergence import RunName, compute_divergence
from stormfork.export.manifest import ExperimentMetadata, export_experiment, verify_manifest
from stormfork.export.quantise import FILE_NAMES
from stormfork.export.reproduction import NULL_FORK_TOLERANCE_KM, null_fork_max_track_diff_km
from stormfork.grid import EARTH_RADIUS_KM
from tests.test_divergence import synthetic_tracks
from tests.test_manifest import example_metadata

# The southernmost track sits at 20 N, where a degree of longitude is longest.
KM_PER_DEGREE_AT_20N = EARTH_RADIUS_KM * math.radians(1) * math.cos(math.radians(20))


def sweep_metadata() -> ExperimentMetadata:
    document = example_metadata().model_dump(mode="json")
    document.update(kind="sweep", perturbation={"type": "relocate", "params": {"east_km": 0}})
    return ExperimentMetadata.model_validate(document)


def sweep_tracks(fork_shift_deg: float) -> dict[RunName, pd.DataFrame]:
    baseline = synthetic_tracks()
    return {
        "baseline": baseline,
        "fork": baseline.assign(lon=baseline.lon + fork_shift_deg),
        "control": synthetic_tracks(0.1),
    }


def write_pngs(directory: Path, runs: list[RunName]) -> None:
    for run in runs:
        (directory / run).mkdir(parents=True, exist_ok=True)
        for file in FILE_NAMES:
            Image.new("L", (1805, 1629), 100).save(directory / run / f"{file}.png")


def write_stops(run_directory: Path, stops: list[dict[str, object]]) -> None:
    """Write stops.json, and tracks moved east by value / 150 degrees for each stop not refused."""
    (run_directory / "stops.json").write_text(json.dumps(stops))
    for stop in stops:
        if stop.get("refused") is None:
            directory = run_directory / str(stop["run"])
            directory.mkdir(parents=True)
            shift = float(str(stop["value"])) / 150
            baseline = synthetic_tracks()
            baseline.assign(lon=baseline.lon + shift).to_csv(directory / "tracks.csv", index=False)
            Image.new("L", (1805, 1629), 100).save(directory / "mslp.png")


ONE_STOP: list[dict[str, object]] = [{"run": "stops/east_km_150", "param": "east_km", "value": 150}]


def exported_sweep(
    tmp_path: Path,
    fork_shift_deg: float,
    kind: str = "sweep",
    stops: list[dict[str, object]] | None = None,
) -> Path:
    run_directory = tmp_path / "run"
    tracks = sweep_tracks(fork_shift_deg)
    write_pngs(run_directory, list(tracks))
    for run, frame in tracks.items():
        frame.to_csv(run_directory / run / "tracks.csv", index=False)
    document = sweep_metadata().model_dump(mode="json")
    document["kind"] = kind
    (run_directory / "metadata.json").write_text(json.dumps(document))
    if kind == "sweep":
        write_stops(run_directory, ONE_STOP if stops is None else stops)
    out = tmp_path / "out"
    export_run_directory(run_directory, out)
    return out


def test_identical_fork_and_baseline_are_zero_apart():
    assert null_fork_max_track_diff_km(sweep_tracks(0)) == 0


def test_the_distance_is_the_largest_member_offset():
    assert null_fork_max_track_diff_km(sweep_tracks(0.05)) == pytest.approx(
        0.05 * KM_PER_DEGREE_AT_20N, rel=1e-3
    )


def test_both_runs_must_cover_the_same_samples():
    tracks = sweep_tracks(0)
    with pytest.raises(ValueError, match="needs both"):
        null_fork_max_track_diff_km({"baseline": tracks["baseline"]})
    tracks["fork"] = synthetic_tracks(members=7)
    with pytest.raises(ValueError, match="do not cover the same samples"):
        null_fork_max_track_diff_km(tracks)


def test_a_sweep_export_records_the_null_fork_distance(tmp_path: Path):
    recorded = verify_manifest(exported_sweep(tmp_path, 0.05)).diagnostics
    assert recorded.null_fork_max_track_diff_km == pytest.approx(
        0.05 * KM_PER_DEGREE_AT_20N, rel=1e-3
    )


def test_other_kinds_leave_the_distance_null(tmp_path: Path):
    out = exported_sweep(tmp_path, 0.05, kind="fork")
    assert verify_manifest(out).diagnostics.null_fork_max_track_diff_km is None


@pytest.mark.parametrize(("shift", "expected"), [(0, 0), (0.05, 0.05 * KM_PER_DEGREE_AT_20N)])
def test_verify_passes_and_reports_a_fork_within_the_tolerance(
    tmp_path: Path, shift: float, expected: float
):
    lines = cli_verify.verify_directory(exported_sweep(tmp_path, shift)).splitlines()
    assert lines[0] == "synthetic_comparison"
    assert lines[-1] == "PASS"
    label, value = lines[-2].split(": ")
    assert label == "null_fork_max_track_diff_km"
    assert float(value) == pytest.approx(expected, abs=1e-3)


def test_verify_rejects_a_fork_beyond_the_tolerance(tmp_path: Path):
    shift = 2 * NULL_FORK_TOLERANCE_KM / KM_PER_DEGREE_AT_20N
    with pytest.raises(ValueError, match=r"over the 10\.0 km limit"):
        cli_verify.verify_directory(exported_sweep(tmp_path, shift))


def test_verify_requires_a_sweep_to_record_the_distance(tmp_path: Path):
    tracks = sweep_tracks(0)
    write_pngs(tmp_path, list(tracks))
    export_experiment(tmp_path, sweep_metadata(), tracks, compute_divergence(tracks))
    with pytest.raises(ValueError, match=r"must record diagnostics\.null_fork_max_track_diff_km"):
        cli_verify.verify_directory(tmp_path)


def test_verify_rejects_a_recorded_distance_the_tracks_do_not_give(tmp_path: Path):
    out = exported_sweep(tmp_path, 0.05)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["diagnostics"]["null_fork_max_track_diff_km"] = 0.0
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="but the tracks give"):
        cli_verify.verify_directory(out)
