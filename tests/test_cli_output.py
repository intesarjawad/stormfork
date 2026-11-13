import json
import math
from pathlib import Path

import pytest

from tests.test_cli import invoke
from tests.test_manifest import example_export

HELP_COMMANDS = [
    (),
    ("scenarios",),
    ("scenarios", "list"),
    ("scenarios", "taper"),
    ("ic",),
    ("ic", "build"),
    ("perturb",),
    ("run",),
    ("track",),
    ("export",),
    ("verify",),
    ("rerun",),
    ("benchmark",),
    ("scores",),
    ("scores", "collect"),
    ("scores", "build"),
]


@pytest.mark.parametrize("command", HELP_COMMANDS)
@pytest.mark.parametrize("columns", ["40", "160"])
def test_all_help_fits_80_columns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
    columns: str,
):
    monkeypatch.setenv("COLUMNS", columns)
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, *command, "--help")
    assert raised.value.code == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert captured.out.startswith("usage: stormfork")
    assert max(map(len, captured.out.splitlines())) <= 80
    assert "\x1b" not in captured.out


def test_root_help_lists_ten_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "--help")
    output = capsys.readouterr().out
    for command in (
        "scenarios",
        "ic",
        "perturb",
        "run",
        "track",
        "export",
        "verify",
        "rerun",
        "benchmark",
        "scores",
    ):
        assert f"    {command}" in output


GOLDEN = """\
synthetic_comparison
step: 4
fork_sep_km: 234.6
control_sep_km: 46.9
paired_p: 0.0039
PASS
"""


@pytest.mark.parametrize("manifest_file", [False, True])
@pytest.mark.parametrize("no_color", [False, True])
def test_verify_output_matches_golden(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    manifest_file: bool,
    no_color: bool,
):
    example_export(tmp_path)
    monkeypatch.chdir(tmp_path)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    invoke(monkeypatch, "verify", "manifest.json" if manifest_file else ".")
    captured = capsys.readouterr()
    assert captured.out == GOLDEN
    assert captured.err == ""
    assert max(map(len, captured.out.splitlines())) <= 80
    assert "\x1b" not in captured.out


def test_verify_reports_the_hero_landfall_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    published = tmp_path / "site/public/experiments/synthetic_comparison"
    published.mkdir(parents=True)
    example_export(published)
    (tmp_path / "site/hero.json").write_text(
        json.dumps({"id": "synthetic_comparison", "landfall_step": 2})
    )
    monkeypatch.chdir(tmp_path)
    invoke(monkeypatch, "verify")
    assert capsys.readouterr().out == GOLDEN.replace("step: 4", "step: 2")
    invoke(monkeypatch, "verify", "synthetic_comparison")
    assert capsys.readouterr().out == GOLDEN.replace("step: 4", "step: 2")


def test_verify_passes_an_adjustment_ratio_at_the_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path, adjustment_ratio=1.5)
    invoke(monkeypatch, "verify", str(tmp_path))
    assert capsys.readouterr().out == GOLDEN


def test_verify_fails_an_adjustment_ratio_above_the_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path, adjustment_ratio=math.nextafter(1.5, 2))
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", str(tmp_path))
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "error: adjustment ratio 1.5000000000000002 is above the 1.5 publication gate\n"
        "next: leave the fork unpublished and record why in docs/demo-gate.md\n"
    )


def test_verify_fails_a_fork_without_a_passed_bounds_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    example_export(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["diagnostics"]["bounds_check"] = None
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", str(tmp_path))
    assert raised.value.code == 1
    assert capsys.readouterr().err == (
        "error: synthetic_comparison records no passed bounds check\n"
        "next: rerun it with stormfork run synthetic_comparison, which checks the edit\n"
    )


def test_a_checkout_copy_without_its_atlases_names_the_missing_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    published = tmp_path / "site/public/experiments/synthetic_comparison"
    published.mkdir(parents=True)
    example_export(published)
    for atlas in published.glob("*/*.png"):
        if atlas.parent.name != "preview":
            atlas.unlink()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "synthetic_comparison")
    assert raised.value.code == 2
    assert capsys.readouterr().err == (
        "error: synthetic_comparison lacks 18 listed files, first baseline/mslp.png\n"
        "next: fetch the atlases with uv run python scripts/fetch_atlases.py\n"
    )
