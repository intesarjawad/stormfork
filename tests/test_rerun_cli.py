import json
import shutil
from argparse import Namespace
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from stormfork import cli_model, cli_rerun, cli_verify
from stormfork.downloads import OFFLINE_ENV
from stormfork.export.manifest import verify_manifest
from stormfork.export.quantise import FILE_NAMES
from stormfork.scenarios import Scenario
from tests.test_cli import invoke
from tests.test_manifest import example_metadata
from tests.test_reproduction import export_tracks, published_tracks

if TYPE_CHECKING:
    import pandas as pd

    from stormfork.export.divergence import RunName

EXPERIMENT = "milton_noise_1pct"
REPOSITORY = Path(__file__).parent.parent
RECIPE = REPOSITORY / "experiments/milton_noise_1pct.yaml"


def write_run_directory(directory: Path, scenario_id: str, lon_shift: float) -> None:
    tracks: dict[RunName, pd.DataFrame] = published_tracks()
    for run, frame in tracks.items():
        (directory / run).mkdir(parents=True)
        frame.assign(lon=frame.lon + lon_shift).to_csv(directory / run / "tracks.csv", index=False)
        for file in FILE_NAMES:
            Image.new("L", (1805, 1629), 100).save(directory / run / f"{file}.png")
    metadata = example_metadata().model_copy(update={"id": scenario_id})
    (directory / "metadata.json").write_text(metadata.model_dump_json())


@pytest.fixture
def stub_forecasts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, Namespace]]:
    calls: list[tuple[str, Namespace]] = []

    def run_forecasts(scenario: Scenario, arguments: Namespace, out: Path) -> None:
        calls.append((scenario.id, arguments))
        write_run_directory(out, scenario.id, 0.01)

    def find_spec(name: str) -> ModuleSpec:
        return ModuleSpec(name, None)

    monkeypatch.setattr(cli_model, "find_spec", find_spec)
    monkeypatch.setattr(cli_rerun, "run_forecasts", run_forecasts)
    shutil.copytree(REPOSITORY / "experiments", tmp_path / "experiments")
    monkeypatch.chdir(tmp_path)
    return calls


def test_a_published_id_goes_through_verify_reproduction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stub_forecasts: list[tuple[str, Namespace]],
):
    published = tmp_path / "published"
    export_tracks(published / EXPERIMENT, published_tracks(), {"id": EXPERIMENT})
    monkeypatch.setattr(cli_verify, "PUBLISHED_DIR", published)
    invoke(monkeypatch, "rerun", EXPERIMENT, "--by", "alice", "--members", "3", "--out", "run")
    rerun_output = capsys.readouterr().out
    assert [(call[0], call[1].members, call[1].steps) for call in stub_forecasts] == [
        (EXPERIMENT, 3, 8)
    ]
    invoke(monkeypatch, "verify", EXPERIMENT, "--reproduction", "run", "--by", "alice")
    assert rerun_output == capsys.readouterr().out
    entry = json.loads(rerun_output)
    assert entry["by"] == "alice"
    assert entry["passed"] is True
    assert 0 < entry["max_track_diff_km"] < 2


def test_a_recipe_file_is_exported_without_a_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stub_forecasts: list[tuple[str, Namespace]],
):
    recipe = tmp_path / "mine.yaml"
    recipe.write_text(RECIPE.read_text())
    # A comparison would look here and find nothing.
    monkeypatch.setattr(cli_verify, "PUBLISHED_DIR", tmp_path / "nothing-published")
    invoke(monkeypatch, "rerun", str(recipe), "--out", "out")
    # Fork member i lies 0.5 (i + 1) degrees east of the baseline's, control member i 0.1 (i + 1).
    assert capsys.readouterr().out == (
        "manifest written to out/manifest.json\n"
        "step: 4\n"
        "fork_sep_km: 234.6\n"
        "control_sep_km: 46.9\n"
        "paired_p: 0.0039\n"
    )
    assert [call[0] for call in stub_forecasts] == [EXPERIMENT]
    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert {
        "manifest.json",
        "tracks.json",
        "tracks.csv",
        "tracks.geojson",
        "divergence.json",
    } <= names
    assert verify_manifest(tmp_path / "out").id == EXPERIMENT


def test_the_default_output_directory_is_named_after_the_experiment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stub_forecasts: list[tuple[str, Namespace]],
):
    recipe = tmp_path / "mine.yaml"
    recipe.write_text(RECIPE.read_text())
    invoke(monkeypatch, "rerun", str(recipe))
    assert capsys.readouterr().out.startswith(f"manifest written to {EXPERIMENT}/manifest.json\n")
    assert stub_forecasts


def test_offline_a_rerun_without_a_local_published_copy_reports_it_was_not_compared(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stub_forecasts: list[tuple[str, Namespace]],
):
    monkeypatch.setattr(cli_verify, "PUBLISHED_DIR", Path("unpublished"))
    invoke(monkeypatch, "rerun", EXPERIMENT, "--by", "alice", "--out", "run")
    captured = capsys.readouterr()
    assert "fetch" not in captured.err
    absent = Path("unpublished") / EXPERIMENT
    expected = (
        "manifest written to run/manifest.json\n"
        f"not compared with the published {EXPERIMENT}: {absent} is absent"
        f" and {OFFLINE_ENV} is set\n"
        f"compare later: stormfork verify {EXPERIMENT} --reproduction run --by alice\n"
    )
    # Long lines wrap at 80 columns, so the words are compared rather than the breaks.
    assert captured.out.split() == expected.split()
    assert captured.out.startswith("manifest written to run/manifest.json\nnot compared")
    assert stub_forecasts
    assert verify_manifest(tmp_path / "run").id == EXPERIMENT


@pytest.mark.usefixtures("stub_forecasts")
def test_a_published_id_needs_the_reproducer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "rerun", EXPERIMENT)
    assert raised.value.code == 2
    assert capsys.readouterr().err == (
        "error: rerunning a published experiment needs --by\nnext: pass your GitHub login to --by\n"
    )
