import sys
from importlib import reload
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from stormfork import cli, cli_model


@pytest.mark.parametrize("command", ["run", "rerun"])
def test_forecast_command_without_the_model_stack_names_the_install_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
):
    for name in ("weathernext", "jax", "haiku"):
        monkeypatch.setitem(sys.modules, name, None)
    reload(cli)
    monkeypatch.setattr(sys, "argv", ["stormfork", command, "storm"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"error: {command} needs the model stack (weathernext), which is not installed\n"
        "next: run uv sync --group model and retry\n"
    )
    assert sys.modules["weathernext"] is None
    assert sys.modules["jax"] is None


@pytest.fixture
def stack_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> ModuleSpec:
        return ModuleSpec(name, None)

    monkeypatch.setattr(cli_model, "find_spec", find_spec)


def fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str) -> str:
    monkeypatch.setattr(sys, "argv", ["stormfork", *args])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    return capsys.readouterr().err


@pytest.mark.usefixtures("stack_present")
def test_rerun_of_a_recipe_from_another_source_names_the_notebook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    error = fail(monkeypatch, capsys, "rerun", "hindcast_milton_2024")
    assert error == (
        "error: hindcast_milton_2024 starts from era5, which rerun cannot fetch\n"
        "next: open the rerun notebook for this recipe\n"
    )


@pytest.mark.usefixtures("stack_present")
def test_rerun_validates_a_recipe_file_and_its_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    original = Path("experiments/milton_noise_1pct.yaml").read_text()
    monkeypatch.chdir(tmp_path)
    recipe = Path("mine.yaml")
    recipe.write_text(original.replace("hres_sample", "era5"))
    assert "starts from era5" in fail(monkeypatch, capsys, "rerun", str(recipe))
    recipe.write_text("id: [unclosed")
    assert "not valid YAML" in fail(monkeypatch, capsys, "rerun", str(recipe))
    recipe.write_text(original)
    error = fail(monkeypatch, capsys, "rerun", str(recipe), "--members", "0")
    assert error.startswith("error: --members and --steps must be at least 1\n")
    assert fail(monkeypatch, capsys, "rerun", str(recipe)) == (
        "error: no cached HRES sample, and STORMFORK_OFFLINE is set\n"
        "next: run stormfork fetch, or pass a local file to --sample\n"
    )


@pytest.mark.usefixtures("stack_present")
def test_unknown_experiment_and_init_are_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    assert fail(monkeypatch, capsys, "run", "missing").startswith('error: unknown experiment "mis')
    error = fail(
        monkeypatch,
        capsys,
        "ic",
        "build",
        "milton_noise_1pct",
        "--init",
        "2024-10-08T00:00",
        "--out",
        "x",
    )
    assert error.startswith("error: milton_noise_1pct has no init 2024-10-08T00:00\n")
