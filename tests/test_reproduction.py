import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from stormfork import cli_verify
from stormfork.errors import UserError
from stormfork.export.divergence import RunName, compute_divergence
from stormfork.export.manifest import HINDCAST_LIMIT, ExperimentMetadata, export_experiment
from stormfork.export.quantise import FILE_NAMES
from stormfork.export.reproduction import (
    Reproduction,
    Reproductions,
    max_track_difference_km,
    tolerance_km,
)
from stormfork.grid import EARTH_RADIUS_KM
from tests.test_cli import invoke
from tests.test_divergence import synthetic_tracks
from tests.test_manifest import example_metadata


def export_tracks(
    directory: Path,
    tracks: dict[RunName, pd.DataFrame],
    changes: dict[str, object] | None = None,
    observed: pd.DataFrame | None = None,
) -> Path:
    for run in tracks:
        (directory / run).mkdir(parents=True)
        for file in FILE_NAMES:
            Image.new("L", (1805, 1629), 100).save(directory / run / f"{file}.png")
    data = {**example_metadata().model_dump(mode="json"), **(changes or {})}
    export_experiment(
        directory,
        ExperimentMetadata.model_validate(data),
        tracks,
        compute_divergence(tracks),
        observed=observed,
    )
    return directory


def published_tracks() -> dict[RunName, pd.DataFrame]:
    return {
        "baseline": synthetic_tracks(),
        "fork": synthetic_tracks(0.5),
        "control": synthetic_tracks(0.1),
    }


def shifted(lon_deg: float, run: RunName = "fork") -> dict[RunName, pd.DataFrame]:
    tracks = published_tracks()
    tracks[run] = tracks[run].assign(lon=tracks[run].lon + lon_deg)
    return tracks


@pytest.fixture
def published(tmp_path: Path) -> Path:
    return export_tracks(tmp_path / "published", published_tracks())


def entry_of(text: str) -> Reproduction:
    return Reproduction.model_validate_json(text)


def test_a_rerun_within_tolerance_passes(tmp_path: Path, published: Path):
    rerun = export_tracks(
        tmp_path / "rerun",
        shifted(0.01),
        {"runtime": {"kind": "cpu", "device": "Darwin arm64"}, "created": "2026-11-06T09:30:00Z"},
    )
    entry = entry_of(cli_verify.reproduce_directory(published, rerun, "alice"))
    # 0.01 degrees of longitude at the lowest latitude, 20 N: R * 0.01 deg * cos(20 deg).
    assert entry.max_track_diff_km == pytest.approx(
        EARTH_RADIUS_KM * math.radians(0.01) * math.cos(math.radians(20)), abs=0.005
    )
    assert entry.passed
    assert entry.by == "alice"
    assert entry.runtime == "cpu: Darwin arm64"
    assert entry.date == date(2026, 11, 6)
    assert entry.code_version == "fixture-v1"


def test_a_rerun_shifted_by_two_grid_cells_fails(tmp_path: Path, published: Path):
    rerun = export_tracks(tmp_path / "rerun", shifted(2.0))
    entry = entry_of(cli_verify.reproduce_directory(published, rerun, "alice"))
    # Haversine for a 2 degree longitude offset at 20 N.
    expected = (
        2 * EARTH_RADIUS_KM * math.asin(math.cos(math.radians(20)) * math.sin(math.radians(1)))
    )
    assert entry.max_track_diff_km == pytest.approx(expected, rel=1e-3)
    assert not entry.passed


def test_the_tolerance_depends_on_whether_the_runtime_kind_changed(tmp_path: Path, published: Path):
    same = export_tracks(tmp_path / "same", shifted(0.2))
    other = export_tracks(
        tmp_path / "other", shifted(0.2), {"runtime": {"kind": "gpu", "device": "L4"}}
    )
    assert not entry_of(cli_verify.reproduce_directory(published, same, "alice")).passed
    assert entry_of(cli_verify.reproduce_directory(published, other, "alice")).passed
    assert tolerance_km("cpu", "cpu") < tolerance_km("cpu", "gpu")


def test_the_ensemble_mean_is_compared_as_well_as_the_members(tmp_path: Path, published: Path):
    # Moving only member 7 east by 1 degree changes its own track by about 105 km
    # and the eight-member mean by an eighth of that, so the larger value wins.
    tracks = published_tracks()
    moved = tracks["control"].copy()
    moved.loc[moved.member == 7, "lon"] += 1.0
    tracks["control"] = moved
    rerun = export_tracks(tmp_path / "rerun", tracks)
    entry = entry_of(cli_verify.reproduce_directory(published, rerun, "alice"))
    member = (
        2 * EARTH_RADIUS_KM * math.asin(math.cos(math.radians(20.7)) * math.sin(math.radians(0.5)))
    )
    assert entry.max_track_diff_km == pytest.approx(member, rel=1e-3)


def test_different_members_runs_or_experiments_are_refused(tmp_path: Path, published: Path):
    fewer = published_tracks()
    fewer["fork"] = synthetic_tracks(0.5, members=7)
    with pytest.raises(ValueError, match="do not cover the same samples"):
        cli_verify.reproduce_directory(published, export_tracks(tmp_path / "fewer", fewer), "alice")
    with pytest.raises(ValueError, match="hold different runs"):
        max_track_difference_km(published_tracks(), {"baseline": synthetic_tracks()})
    hindcast = export_tracks(
        tmp_path / "hindcast",
        {"baseline": synthetic_tracks()},
        {
            "kind": "hindcast",
            "id": "synthetic_comparison-2024100700",
            "parent_id": "synthetic_comparison",
            "limits": [HINDCAST_LIMIT],
        },
        observed=synthetic_tracks(),
    )
    with pytest.raises(ValueError, match='holds "synthetic_comparison-2024100700"'):
        cli_verify.reproduce_directory(published, hindcast, "alice")
    other = export_tracks(tmp_path / "other", published_tracks(), {"id": "another"})
    with pytest.raises(ValueError, match='holds "another", not "synthetic_comparison"'):
        cli_verify.reproduce_directory(published, other, "alice")


def test_an_unexported_run_directory_names_the_next_step(tmp_path: Path, published: Path):
    (tmp_path / "run").mkdir()
    with pytest.raises(UserError, match=r"has no manifest\.json"):
        cli_verify.reproduce_directory(published, tmp_path / "run", "alice")


def test_a_tampered_rerun_is_refused(tmp_path: Path, published: Path):
    rerun = export_tracks(tmp_path / "rerun", shifted(0.01))
    (rerun / "tracks.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match the manifest"):
        cli_verify.reproduce_directory(published, rerun, "alice")


def test_verify_prints_one_json_entry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    published: Path,
):
    rerun = export_tracks(tmp_path / "rerun", shifted(0.01))
    invoke(monkeypatch, "verify", str(published), "--reproduction", str(rerun), "--by", "alice")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert list(json.loads(lines[0])) == [
        "by",
        "runtime",
        "date",
        "code_version",
        "max_track_diff_km",
        "passed",
    ]
    assert Reproductions.validate_json(f"[{lines[0]}]")[0].passed


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--by", "alice"], "--reproduction and --by must be supplied together"),
        (["--reproduction", "x"], "--reproduction and --by must be supplied together"),
        (["--reproduction", "x", "--by", "not a user"], "is not a GitHub user name"),
        (["--reproduction", "x", "--by", "alice-"], "is not a GitHub user name"),
    ],
)
def test_verify_checks_the_reproduction_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    published: Path,
    arguments: list[str],
    message: str,
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", str(published), *arguments)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry",
    [
        {"by": "alice", "runtime": "cpu: x", "date": "2026-11-06", "code_version": "v1"},
        {"by": "a b", "runtime": "cpu", "date": "2026-11-06", "code_version": "v1"}
        | {"max_track_diff_km": 1, "passed": True},
        {"by": "alice", "runtime": "cpu", "date": "yesterday", "code_version": "v1"}
        | {"max_track_diff_km": 1, "passed": True},
        {"by": "alice", "runtime": "cpu", "date": "2026-11-06", "code_version": "v1"}
        | {"max_track_diff_km": -1, "passed": True},
        {"by": "alice", "runtime": "cpu", "date": "2026-11-06", "code_version": "v1"}
        | {"max_track_diff_km": 1, "passed": True, "extra": 1},
    ],
)
def test_ledger_entries_are_validated(entry: dict[str, object]):
    with pytest.raises(ValueError, match="validation error"):
        Reproductions.validate_python([entry])
