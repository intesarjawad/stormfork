import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import BaseModel, Field

from stormfork.scores.fetch import collect_backfill
from tests.scores.test_previous_runs import TOWN
from tests.test_pr_author import commit, git

ROOT = Path(__file__).resolve().parents[1]


class Step(BaseModel):
    name: str = ""
    run: str = ""
    uses: str = ""


class Job(BaseModel):
    steps: list[Step] = Field(default_factory=list[Step])


class Workflow(BaseModel):
    jobs: dict[str, Job]


def steps(name: str, job: str) -> list[Step]:
    return (
        Workflow.model_validate(
            yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
        )
        .jobs[job]
        .steps
    )


def recorded_request(url: str) -> bytes:
    name = "previous_runs.json" if "previous-runs-api" in url else "asos_chunk.csv"
    return (ROOT / "tests/fixtures" / name).read_bytes()


def run_step(step: Step, checkout: Path, runner_temp: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["bash", "-c", step.run],  # noqa: S607
        cwd=checkout,
        env={**os.environ, "RUNNER_TEMP": str(runner_temp)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_pages_requires_verified_atlases_before_building(tmp_path: Path):
    build = steps("pages", "build")
    fetch = next(step for step in build if "fetch_atlases.py" in step.run)
    assert build.index(fetch) < next(
        index for index, step in enumerate(build) if "pnpm run build" in step.run
    )
    assert any("setup-uv" in step.uses for step in build[: build.index(fetch)])
    assert any("uv sync --frozen" in step.run for step in build[: build.index(fetch)])
    result = run_step(fetch, tmp_path, tmp_path)
    assert result.returncode != 0
    assert "atlas" in result.stderr.lower()
    assert "lock" in result.stderr.lower()


def test_fresh_runner_restores_backfill_before_selecting_the_next_chunk(tmp_path: Path):
    collect = steps("scoreboard", "collect")
    backfill_index = next(
        index for index, step in enumerate(collect) if "scores backfill" in step.run
    )
    restore = next(
        step for step in collect[:backfill_index] if step.name == "Restore archived backfill chunks"
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    git(archive, "init", "-q", "--initial-branch=data")
    start = datetime(2026, 11, 4, tzinfo=UTC)
    end = datetime(2027, 2, 4, tzinfo=UTC)
    first = collect_backfill([TOWN], archive, since=start, until=end, request=recorded_request)
    assert first is not None
    commit(archive, "first chunk")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    git(checkout, "remote", "add", "origin", str(archive))
    runner = tmp_path / "fresh-runner"
    runner.mkdir()
    result = run_step(restore, checkout, runner)
    assert result.returncode == 0, result.stderr
    second = collect_backfill(
        [TOWN], runner / "backfill", since=start, until=end, request=recorded_request
    )
    assert second is not None
    assert second.start > first.end
    for folder in ["previous_runs", "observations"]:
        assert (runner / "backfill" / folder / f"{first.start}.parquet").read_bytes() == (
            archive / folder / f"{first.start}.parquet"
        ).read_bytes()


@pytest.mark.parametrize("data_branch", [False, True])
def test_backfill_restore_accepts_an_empty_archive(tmp_path: Path, *, data_branch: bool):
    restore = next(
        step
        for step in steps("scoreboard", "collect")
        if step.name == "Restore archived backfill chunks"
    )
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "--initial-branch=data" if data_branch else "--initial-branch=main")
    git(origin, "commit", "--allow-empty", "-m", "empty archive")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    git(checkout, "remote", "add", "origin", str(origin))
    result = run_step(restore, checkout, tmp_path / "runner")
    assert result.returncode == 0, result.stderr


def test_backfill_restore_stops_when_the_archive_cannot_be_reached(tmp_path: Path):
    restore = next(
        step
        for step in steps("scoreboard", "collect")
        if step.name == "Restore archived backfill chunks"
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "remote", "add", "origin", str(tmp_path / "missing-origin"))
    result = run_step(restore, tmp_path, tmp_path / "runner")
    assert result.returncode != 0
    assert "Could not read from remote repository" in result.stderr


def test_scoring_and_deployment_can_run_after_collection_fails():
    document = yaml.safe_load((ROOT / ".github/workflows/scoreboard.yml").read_text())
    score = document["jobs"]["score"]
    assert score["needs"] == "collect"
    assert "always()" in score["if"]
    assert "!cancelled()" in score["if"]
    fresh = next(step for step in score["steps"] if "scores fetch" in step.get("run", ""))
    assert fresh["continue-on-error"] is True
    deploy = document["jobs"]["deploy"]
    assert "always()" in deploy["if"]
    assert "needs.score.result == 'success'" in deploy["if"]


@pytest.mark.parametrize("fresh", [False, True])
def test_board_build_uses_complete_stored_inputs_when_the_fetch_fails(
    tmp_path: Path, *, fresh: bool
):
    runner = tmp_path / "runner"
    stored = runner / "data/score-inputs"
    stored.mkdir(parents=True)
    (runner / "data/wn2").mkdir()
    pd.DataFrame(
        [
            {
                "town": "KMIA",
                "variable": "temperature_2m",
                "init_time": datetime(2026, 11, 4, tzinfo=UTC),
                "valid_time": datetime(2026, 11, 4, tzinfo=UTC),
                "ens_mean": 21.0,
                "member_0": 21.2,
            }
        ]
    ).to_parquet(runner / "data/wn2/run.parquet", index=False)
    for directory in [stored, runner / "score-inputs"] if fresh else [stored]:
        directory.mkdir(exist_ok=True)
        scoring_date = "2026-11-06" if directory == runner / "score-inputs" else "2026-11-05"
        (directory / "scoring-day.txt").write_text(f"{scoring_date}\n")
        observed = 22.5 if directory == runner / "score-inputs" else 21.5
        (directory / "observations.csv").write_text(
            f"town,variable,valid_time,value\nKMIA,temperature_2m,2026-11-04T00:00:00Z,{observed}\n"
        )
        (directory / "ifs.csv").write_text(
            "town,variable,valid_time,lead_days,value\nKMIA,temperature_2m,2026-11-04T00:00:00Z,0,20.5\n"
        )
    rebuild = next(
        step for step in steps("scoreboard", "score") if step.name == "Rebuild scores.json"
    )
    script = rebuild.run.replace("${{ steps.wn2_probe.outputs.available }}", "")
    script = script.replace("${{ steps.score_inputs.outcome }}", "success" if fresh else "failure")
    result = run_step(rebuild.model_copy(update={"run": script}), ROOT, runner)
    assert result.returncode == 0, result.stderr
    board = json.loads((runner / "scores.json").read_text())
    ifs_cell = next(
        cell for cell in board["cells"] if cell["model"] == "IFS" and cell["lead_days"] == 0
    )
    assert ifs_cell["mae"] == (2.0 if fresh else 1.0)
    assert board["generated_at"] == ("2026-11-06" if fresh else "2026-11-05")
    assert board["stale_since"] == ("2026-11-08" if fresh else "2026-11-07")
    assert ("Using stored score inputs" in result.stdout) is not fresh


def test_board_build_stops_without_complete_fresh_or_stored_inputs(tmp_path: Path):
    runner = tmp_path / "runner"
    (runner / "data/score-inputs").mkdir(parents=True)
    (runner / "data/score-inputs/observations.csv").write_text("partial")
    rebuild = next(
        step for step in steps("scoreboard", "score") if step.name == "Rebuild scores.json"
    )
    script = rebuild.run.replace("${{ steps.wn2_probe.outputs.available }}", "")
    script = script.replace("${{ steps.score_inputs.outcome }}", "failure")
    result = run_step(rebuild.model_copy(update={"run": script}), ROOT, runner)
    assert result.returncode != 0
    assert not (runner / "scores.json").exists()
    assert "No complete stored score inputs" in result.stderr


@pytest.mark.parametrize("fresh", [False, True])
def test_score_storage_replaces_inputs_only_after_a_complete_fetch(tmp_path: Path, *, fresh: bool):
    runner = tmp_path / "runner"
    current = runner / "score-inputs"
    stored = runner / "data/score-inputs"
    current.mkdir(parents=True)
    stored.mkdir(parents=True)
    files = ("observations.csv", "ifs.csv", "scoring-day.txt")
    for name in files:
        (current / name).write_text(f"new {name}")
        (stored / name).write_text(f"stored {name}")
    (runner / "scores.json").write_text("{}")
    store = next(
        step
        for step in steps("scoreboard", "score")
        if step.name == "Commit scores.json to the data branch"
    )
    script = "git() { return 0; }\n" + store.run.replace(
        "${{ steps.score_inputs.outcome }}", "success" if fresh else "failure"
    )
    result = run_step(store.model_copy(update={"run": script}), ROOT, runner)
    assert result.returncode == 0, result.stderr
    assert {name: (stored / name).read_text() for name in files} == {
        name: f"{'new' if fresh else 'stored'} {name}" for name in files
    }
