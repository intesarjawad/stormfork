import ast
import re
import shlex
from pathlib import Path
from typing import Any

import nbformat
import pytest

from stormfork.cli import build_parser
from stormfork.scenarios import Scenario, load_scenarios

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "rerun_experiment.ipynb"
VARIABLE = re.compile(r"\$(\w+)")
SAMPLES = {
    "INIT": "2024-10-07T00:00:00",
    "RUNTIME": "cpu",
    "PERTURB_ARGS": (
        "--operation noise --centre-lat 0 --centre-lon 0 --radius-km 400 --amplitude 0.01 --seed 0"
    ),
    "TARGET_ARGS": "--target-track-id sid",
    "FORK_TRACK_ARGS": "--target-track-id sid --moved-lat 1 --moved-lon 2",
}


def code_cells() -> list[str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def stormfork_commands(source: str) -> list[str]:
    return [
        line.removeprefix("!").strip()
        for line in source.splitlines()
        if line.startswith("!stormfork ")
    ]


def assigned_names(source: str) -> set[str]:
    cleaned = "\n".join(line for line in source.splitlines() if not line.startswith(("!", "%")))
    return {
        target.id
        for node in ast.walk(ast.parse(cleaned))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_every_stormfork_command_parses_with_the_cli():
    commands = [command for cell in code_cells() for command in stormfork_commands(cell)]
    assert [shlex.split(command)[1] for command in commands] == [
        "fetch",
        "scenarios",
        "ic",
        "perturb",
        "run",
        "track",
        "track",
        "track",
        "export",
        "verify",
    ]
    for command in commands:
        expanded = VARIABLE.sub(lambda found: SAMPLES.get(found.group(1), "x"), command)
        build_parser().parse_args(shlex.split(expanded)[1:])


def test_every_variable_is_set_by_an_earlier_cell():
    defined: set[str] = set()
    for source in code_cells():
        used = {
            name
            for line in source.splitlines()
            if line.startswith("!")
            for name in VARIABLE.findall(line)
        }
        assert used <= defined, used - defined
        defined |= assigned_names(source)


def test_the_last_cell_verifies_the_reproduction():
    last = code_cells()[-1]
    assert stormfork_commands(last) == [
        "stormfork verify $EXPERIMENT_ID --reproduction $RUN_DIR --by $GITHUB_USER"
    ]


def edit_cell() -> str:
    return next(source for source in code_cells() if "PERTURB_ARGS = " in source)


@pytest.mark.parametrize(
    "scenario",
    [
        item
        for item in load_scenarios(NOTEBOOK.parent.parent / "experiments")
        if item.kind == "fork"
    ],
    ids=lambda item: item.id,
)
def test_the_edit_cell_builds_arguments_the_cli_accepts(
    monkeypatch: pytest.MonkeyPatch, scenario: Scenario
):
    monkeypatch.chdir(NOTEBOOK.parent.parent)
    namespace: dict[str, Any] = {"EXPERIMENT_ID": scenario.id}
    exec(edit_cell(), namespace)  # noqa: S102
    perturb = build_parser().parse_args(
        ["perturb", "input.nc", "--dry-run", *shlex.split(namespace["PERTURB_ARGS"])]
    )
    assert scenario.edit is not None
    assert perturb.operation == scenario.edit.type
    track = build_parser().parse_args(
        [
            *("track", "f.nc", "--ibtracs", "i.nc", "--init", namespace["INIT"]),
            *("--out", "t.csv", *shlex.split(namespace["FORK_TRACK_ARGS"])),
        ]
    )
    assert track.target_track_id == scenario.storm
    assert (track.moved_lat is not None) == (scenario.edit.type == "relocate")


def test_the_edit_cell_refuses_other_kinds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(NOTEBOOK.parent.parent)
    with pytest.raises(ValueError, match="this notebook reruns forks"):
        exec(edit_cell(), {"EXPERIMENT_ID": "hindcast_milton_2024"})  # noqa: S102
