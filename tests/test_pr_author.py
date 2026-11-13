import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "verify_manifest.py"
EXPERIMENT = Path("site/public/experiments/milton_noise_1pct")


def entry(by: str, runtime: str = "cpu: Darwin arm64") -> dict[str, object]:
    return {
        "by": by,
        "runtime": runtime,
        "date": "2026-11-06",
        "code_version": "v1",
        "max_track_diff_km": 0.5,
        "passed": True,
    }


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments],  # noqa: S607
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit(repository: Path, message: str) -> str:
    git(repository, "add", "-A")
    git(repository, "commit", "-m", message)
    return git(repository, "rev-parse", "HEAD")


def write_ledger(repository: Path, entries: list[dict[str, object]]) -> None:
    directory = repository / EXPERIMENT
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reproductions.json").write_text(json.dumps(entries))


def check(repository: Path, author: str, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--pr-author", author, "--base", base],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    return tmp_path


def test_an_added_entry_must_name_the_pull_request_author(repository: Path):
    write_ledger(repository, [entry("carol")])
    base = commit(repository, "first reproduction")
    write_ledger(repository, [entry("carol"), entry("bob", "gpu: L4")])
    commit(repository, "second reproduction")
    failed = check(repository, "alice", base)
    assert failed.returncode == 1
    assert "added entry is by 'bob', not the author 'alice'" in failed.stderr
    write_ledger(repository, [entry("carol"), entry("alice", "gpu: L4")])
    commit(repository, "fixed")
    assert check(repository, "alice", base).returncode == 0


def test_logins_compare_without_regard_to_case(repository: Path):
    write_ledger(repository, [])
    base = commit(repository, "empty ledger")
    write_ledger(repository, [entry("Alice")])
    commit(repository, "reproduction")
    assert check(repository, "alice", base).returncode == 0


def test_entries_already_on_the_base_are_ignored(repository: Path):
    write_ledger(repository, [entry("bob")])
    base = commit(repository, "bob reproduced it")
    (repository / "README.md").write_text("unrelated\n")
    commit(repository, "unrelated change")
    result = check(repository, "alice", base)
    assert result.returncode == 0
    assert result.stderr == ""


def test_a_ledger_new_since_the_base_is_all_added(repository: Path):
    (repository / "README.md").write_text("x\n")
    base = commit(repository, "nothing yet")
    write_ledger(repository, [entry("bob")])
    commit(repository, "bob reproduced it")
    assert check(repository, "alice", base).returncode == 1


def test_a_duplicate_of_an_existing_entry_counts_as_added(repository: Path):
    write_ledger(repository, [entry("bob")])
    base = commit(repository, "bob reproduced it")
    write_ledger(repository, [entry("bob"), entry("bob")])
    commit(repository, "again")
    assert check(repository, "alice", base).returncode == 1


def test_a_new_experiment_must_be_run_by_the_author(repository: Path):
    (repository / "README.md").write_text("x\n")
    base = commit(repository, "nothing yet")
    directory = repository / EXPERIMENT
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"id": "milton_noise_1pct", "run_by": "bob"})
    )
    commit(repository, "new experiment")
    failed = check(repository, "alice", base)
    assert failed.returncode == 1
    assert "run_by is 'bob', not the author 'alice'" in failed.stderr
    (directory / "manifest.json").write_text(
        json.dumps({"id": "milton_noise_1pct", "run_by": "alice"})
    )
    commit(repository, "fixed")
    assert check(repository, "alice", base).returncode == 0


def test_an_unchanged_run_by_is_not_rechecked(repository: Path):
    directory = repository / EXPERIMENT
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"id": "milton_noise_1pct", "run_by": "bob"})
    )
    base = commit(repository, "bob ran it")
    (directory / "title.txt").write_text("x")
    commit(repository, "unrelated")
    assert check(repository, "alice", base).returncode == 0


def test_an_invalid_entry_fails_validation_without_a_base(repository: Path):
    write_ledger(repository, [{**entry("alice"), "passed": "maybe"}])
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT)], cwd=repository, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "passed" in result.stderr


def test_an_unknown_base_is_an_error(repository: Path):
    write_ledger(repository, [entry("alice")])
    commit(repository, "reproduction")
    failed = check(repository, "alice", "origin/missing")
    assert failed.returncode == 1
    assert "origin/missing is not a commit" in failed.stderr
