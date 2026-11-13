import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.report_data_faults import LABEL, TITLE, main


class RecordedGh:
    """A gh stand-in that records each call and answers the issue list."""

    def __init__(self, open_issues: Sequence[int]) -> None:
        self.open_issues = list(open_issues)
        self.calls: list[tuple[str, ...]] = []
        self.bodies: list[str] = []

    def __call__(self, arguments: Sequence[str]) -> str:
        self.calls.append(tuple(arguments))
        if arguments[:2] == ("issue", "list"):
            return json.dumps([{"number": number} for number in self.open_issues])
        if "--body-file" in arguments:
            body_path = Path(arguments[arguments.index("--body-file") + 1])
            self.bodies.append(body_path.read_text(encoding="utf-8"))
        return ""


def write_scores(path: Path, *, quarantined: bool) -> None:
    cells = [
        {
            "town": "KMIA",
            "variable": "temperature_2m",
            "lead_days": 0,
            "model": "WN2 mean",
            "mae": 0.8,
            "sample_size": 100,
            "blocks": 13,
            "statistic": None,
            "p_value": None,
            "mean_difference": 0.1,
            "block_sd": 0.0,
            "quarantined": quarantined,
            "rejected": False,
            "interval": None,
            "badge": None,
        }
    ]
    path.write_text(
        json.dumps({"generated_at": "2026-11-07", "cells": cells}) + "\n", encoding="utf-8"
    )


def test_degenerate_cells_without_an_open_issue_create_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    scores = tmp_path / "scores.json"
    write_scores(scores, quarantined=True)
    gh = RecordedGh([])
    monkeypatch.setattr("scripts.report_data_faults.run_gh", gh)

    assert main([str(scores)]) == 0

    assert [call[:2] for call in gh.calls] == [("issue", "list"), ("issue", "create")]
    create = gh.calls[1]
    assert "--label" in create
    assert create[create.index("--label") + 1] == LABEL
    assert "--title" in create
    assert create[create.index("--title") + 1] == TITLE
    assert "KMIA" in gh.bodies[0]
    assert "temperature_2m" in gh.bodies[0]


def test_degenerate_cells_with_an_open_issue_comment_and_do_not_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    scores = tmp_path / "scores.json"
    write_scores(scores, quarantined=True)
    gh = RecordedGh([42])
    monkeypatch.setattr("scripts.report_data_faults.run_gh", gh)

    assert main([str(scores)]) == 0

    assert [call[:2] for call in gh.calls] == [("issue", "list"), ("issue", "comment")]
    comment = gh.calls[1]
    assert comment[2] == "42"
    assert "KMIA" in gh.bodies[0]
    assert all(call[:2] != ("issue", "create") for call in gh.calls)


def test_no_degenerate_cells_make_no_gh_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    scores = tmp_path / "scores.json"
    write_scores(scores, quarantined=False)
    gh = RecordedGh([])
    monkeypatch.setattr("scripts.report_data_faults.run_gh", gh)

    assert main([str(scores)]) == 0

    assert gh.calls == []
