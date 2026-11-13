import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from stormfork.scores.fetch import collect_wn2
from tests.scores.test_fetch_wn2 import FIXTURES, INIT, TOWNS
from tests.test_cli import invoke
from tests.test_manifest import example_export


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("unknown",),
        ("scenarios",),
        ("scenarios", "taper", "storm"),
        ("track", "missing.nc"),
        ("perturb", "missing.nc", "--dry-run"),
        ("run", "storm", "--runtime", "invalid"),
        ("verify", "--unknown"),
        ("run",),
        ("ic", "build", "storm"),
        ("export", "run-dir"),
        ("rerun",),
    ],
)
def test_usage_errors_have_two_lines_and_one_next_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, *arguments)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    lines = captured.err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("error: ")
    assert lines[1].startswith("next: run stormfork")
    assert "--help" in lines[1]
    assert max(map(len, lines)) <= 80


def test_export_says_which_command_writes_the_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "export", "run", "--out", "out")
    assert raised.value.code == 2
    assert capsys.readouterr().err == (
        'error: "run" has no metadata.json\nnext: write it with stormfork run <id> --out run\n'
    )


def test_scores_build_requires_its_inputs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "scores", "build")
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    lines = captured.err.splitlines()
    assert len(lines) == 2
    assert "arguments" in lines[0]
    assert lines[1].startswith("next: run stormfork")
    assert max(map(len, lines)) <= 80


def test_verify_modified_artifact_fails_without_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path)
    (tmp_path / "tracks.csv").write_bytes(b"changed")
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", str(tmp_path))
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("error: tracks.csv SHA-256 ")
    assert lines[1] == "next: check the reported data or path and retry"
    assert max(map(len, lines)) <= 80


def test_verify_rejects_a_divergence_that_does_not_follow_from_the_tracks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path)
    divergence = json.loads((tmp_path / "divergence.json").read_text())
    divergence["steps"][3]["fork_sep_km"] += 1e-6
    data = (json.dumps(divergence) + "\n").encode()
    (tmp_path / "divergence.json").write_bytes(data)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["divergence"]["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", str(tmp_path))
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines()[0].startswith("error: divergence.json differs from ")
    assert "divergence.steps[3].fork_sep_km" in captured.err


def test_verify_accepts_a_difference_below_the_tolerance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path)
    divergence = json.loads((tmp_path / "divergence.json").read_text())
    divergence["steps"][3]["fork_sep_km"] += 1e-10
    data = (json.dumps(divergence) + "\n").encode()
    (tmp_path / "divergence.json").write_bytes(data)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["divergence"]["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    invoke(monkeypatch, "verify", str(tmp_path))
    assert capsys.readouterr().out.endswith("PASS\n")


def test_debug_retains_the_original_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        invoke(monkeypatch, "--debug", "verify", str(tmp_path))


def test_missing_taper_row_names_storm_and_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            "scenarios",
            "taper",
            "missing",
            "--ibtracs",
            "tests/fixtures/ibtracs_katrina_taper.csv",
            "--init",
            "2005-08-24T12:00:00",
        )
    assert raised.value.code == 2
    assert "missing at 2005-08-24T12:00:00" in capsys.readouterr().err


def test_stale_collection_reports_failure_and_a_timed_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    def stale_collection(*_args: object, **_kwargs: object) -> Path:
        return collect_wn2(
            TOWNS,
            tmp_path,
            now=INIT + timedelta(days=2),
            request=lambda _url: (FIXTURES / "wn2_meta.json").read_bytes(),
        )

    monkeypatch.setattr("stormfork.cli.collect_wn2", stale_collection)
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "scores", "collect", "--output", str(tmp_path))
    assert raised.value.code == 3
    captured = capsys.readouterr()
    assert not captured.out
    assert "stale" in captured.err
    assert captured.err.splitlines()[1] == (
        "next: retry this command in 30 minutes; see docs/scores.md for metadata checks"
    )
    assert not list(tmp_path.rglob("*.parquet"))
