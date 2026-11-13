"""A sweep's stops reach divergence.json with Holm-adjusted p values, and the manifest's slider."""

import hashlib
import json
from pathlib import Path

import pytest

from stormfork.cli_export import STOPS_FILE, export_run_directory
from stormfork.errors import UserError
from stormfork.export.divergence import compute_divergence
from stormfork.export.manifest import Manifest, manifest_paths, verify_manifest
from stormfork.export.sweep import StopRun, stop_name, sweep_evidence
from tests.test_divergence import synthetic_tracks
from tests.test_sweep_null_fork import exported_sweep, sweep_tracks

STOPS: list[dict[str, object]] = [
    {"run": "stops/east_km_-150", "param": "east_km", "value": -150, "refused": "land"},
    {"run": "stops/east_km_300", "param": "east_km", "value": 300, "adjustment_ratio": 1.2},
    {"run": "stops/east_km_150", "param": "east_km", "value": 150},
    {"run": "stops/east_km_450", "param": "east_km", "value": 450, "adjustment_ratio": 1.6},
]


def moved(degrees: float) -> StopRun:
    baseline = synthetic_tracks()
    return StopRun("stops/x", degrees, baseline.assign(lon=baseline.lon + degrees), b"")


def test_each_stop_is_holm_adjusted_across_the_non_null_stops_at_every_step():
    # Every member of both stops lies further from the baseline than the control: p = 1/256 each.
    tracks = sweep_tracks(0.05)
    divergence, _ = sweep_evidence(
        "delta_k", 0, tracks, compute_divergence(tracks), [moved(1.0), moved(2.0)]
    )
    assert list(divergence.stops) == ["delta_k_0", "delta_k_1", "delta_k_2"]
    null, stop = divergence.stops["delta_k_0"], divergence.stops["delta_k_2"]
    assert divergence.steps == null
    assert [row.paired_p for row in stop] == [1 / 256] * 5
    assert [row.paired_p_holm for row in stop] == [2 / 256] * 5
    assert [row.paired_p for row in null] == [1.0] * 5
    assert [row.paired_p_holm for row in null] == [None] * 5


def test_a_sweep_over_another_parameter_publishes_statistics_without_a_slider():
    tracks = sweep_tracks(0)
    _, sweep = sweep_evidence("factor", 1.0, tracks, compute_divergence(tracks), [moved(0.5)])
    assert sweep is None


def test_stops_are_named_by_direction_and_distance():
    assert stop_name("east_km", -300) == "west_300"
    assert stop_name("north_km", 150.5) == "north_150.5"
    assert stop_name("east_km", 0) == "none_0"
    assert stop_name("delta_k", -2) == "delta_k_-2"


def test_export_publishes_each_stop_the_check_and_gate_pass(tmp_path: Path):
    out = exported_sweep(tmp_path, 0.05, stops=STOPS)
    manifest = verify_manifest(out)
    assert manifest.sweep is not None
    assert manifest.sweep.name == "east_km"
    assert manifest.sweep.baseline == "none_0"
    assert [stop.id for stop in manifest.sweep.stops] == ["none_0", "east_150", "east_300"]
    null, east = manifest.sweep.stops[0], manifest.sweep.stops[1]
    assert (null.tracks, null.atlas, null.divergence, null.sha256) == (
        "tracks.json",
        "fork/mslp.png",
        "divergence.json",
        {},
    )
    assert (east.direction, east.distance_km, east.atlas) == (
        "east",
        150,
        "stops/east_km_150/mslp.png",
    )
    assert east.sha256 == {
        path: hashlib.sha256((out / path).read_bytes()).hexdigest()
        for path in (east.tracks, east.atlas, east.divergence)
    }
    assert {east.tracks, east.atlas, east.divergence} <= manifest_paths(manifest)
    divergence = json.loads((out / "divergence.json").read_text())
    assert sorted(divergence["stops"]) == ["east_150", "east_300", "none_0"]
    assert divergence["stops"]["east_150"][0]["paired_p_holm"] == 2 / 256
    stop_divergence = json.loads((out / east.divergence).read_text())
    assert stop_divergence["steps"] == divergence["stops"]["east_150"]
    runs = {sample["run"] for sample in json.loads((out / east.tracks).read_text())}
    assert runs == {"baseline", "fork", "control"}


def test_a_changed_stop_file_fails_verification(tmp_path: Path):
    out = exported_sweep(tmp_path, 0.05, stops=STOPS)
    (out / "stops/east_km_150/tracks.json").write_text("[]\n")
    with pytest.raises(ValueError, match=r"stops/east_km_150/tracks\.json SHA-256"):
        verify_manifest(out)


def test_the_manifest_hashes_every_file_a_stop_names(tmp_path: Path):
    document = json.loads((exported_sweep(tmp_path, 0.05) / "manifest.json").read_text())
    document["sweep"]["stops"][1]["sha256"] = {}
    with pytest.raises(ValueError, match="sweep stop east_150 names a file the manifest does not"):
        Manifest.model_validate(document)
    document["kind"] = "fork"
    with pytest.raises(ValueError, match="only a sweep publishes sweep stops"):
        Manifest.model_validate(document)


def test_a_published_stop_needs_its_tracks(tmp_path: Path):
    exported_sweep(tmp_path, 0)
    (tmp_path / "run/stops/east_km_150/tracks.csv").unlink()
    with pytest.raises(UserError, match=r"stops/east_km_150/tracks\.csv does not exist"):
        export_run_directory(tmp_path / "run", tmp_path / "out")


def test_a_sweep_without_stops_names_the_run_command(tmp_path: Path):
    exported_sweep(tmp_path, 0)
    (tmp_path / "run" / STOPS_FILE).unlink()
    with pytest.raises(UserError, match=r"has no stops\.json") as raised:
        export_run_directory(tmp_path / "run", tmp_path / "out")
    assert raised.value.next_step.startswith("write it with stormfork run <id> --out")
