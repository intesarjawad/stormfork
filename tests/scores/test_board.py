from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stormfork.scores.board import STALE_AFTER_DAYS, Scoreboard, ScoreCell, build_scoreboard
from stormfork.scores.build import (
    build_from_files,
    load_ifs,
    load_observations,
    load_wn2,
    write_scoreboard,
)
from stormfork.scores.metrics import (
    BLOCK_DAYS,
    SLOTS_PER_BLOCK,
    STEP_HOURS,
    WINDOW_DAYS,
    IfsForecast,
    Observation,
    Wn2Run,
)

SCORING_DAY = datetime(2026, 11, 5, tzinfo=UTC)
WINDOW_START = SCORING_DAY - timedelta(days=91)
TOWN = "KMIA"
VARIABLE = "temperature_2m"
SLOTS = SLOTS_PER_BLOCK // 2


def _board(
    series: Mapping[str, Sequence[Sequence[float]]], *, ifs_error: float = 10.0
) -> tuple[list[Observation], list[IfsForecast], list[Wn2Run]]:
    observations: list[Observation] = []
    ifs: list[IfsForecast] = []
    wn2: list[Wn2Run] = []
    for town, blocks in series.items():
        for block_index, differences in enumerate(blocks):
            for slot_index, difference in enumerate(differences):
                valid_time = WINDOW_START + timedelta(
                    days=BLOCK_DAYS * block_index, hours=STEP_HOURS * slot_index
                )
                observations.append(Observation(town, VARIABLE, valid_time, 0.0))
                ifs.append(IfsForecast(town, VARIABLE, valid_time, 0, ifs_error))
                value = ifs_error + difference
                wn2.append(
                    Wn2Run(
                        town=town,
                        variable=VARIABLE,
                        init_time=valid_time,
                        valid_time=valid_time,
                        ens_mean=value,
                        member_0=value,
                    )
                )
    return observations, ifs, wn2


def _constant_blocks(means: Sequence[float]) -> dict[str, list[list[float]]]:
    return {TOWN: [[mean] * SLOTS for mean in means]}


def _mean_cell(board: Scoreboard) -> ScoreCell:
    return next(cell for cell in board.cells if cell.model == "WN2 mean")


def test_board_has_one_cell_per_model_and_town():
    observations, ifs, wn2 = _board(_constant_blocks([0.1] * 13))

    board = build_scoreboard(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    assert board.generated_at == SCORING_DAY.date()
    assert board.stale_since == SCORING_DAY.date() + timedelta(days=STALE_AFTER_DAYS)
    assert board.window_days == WINDOW_DAYS
    assert board.lead_days == [0]
    assert board.variables == [VARIABLE]
    assert board.models == ["IFS", "WN2 mean", "WN2 member 0"]
    assert board.towns == [TOWN]
    assert len(board.cells) == 3
    mean = _mean_cell(board)
    assert mean.mae == pytest.approx(10.1)
    assert mean.blocks == 13
    assert mean.badge is None
    ifs_cell = next(cell for cell in board.cells if cell.model == "IFS")
    assert ifs_cell.blocks == 0
    assert ifs_cell.interval is None


def test_published_comparators_share_counts_and_include_physical_units():
    observations, ifs, wn2 = _board(_constant_blocks([0.1] * 13))
    board = build_scoreboard(observations, ifs, wn2[:20], scoring_day=SCORING_DAY, lead_days=[0])
    assert {cell.sample_size for cell in board.cells} == {20}
    assert {cell.model_dump()["unit"] for cell in board.cells} == {"°C"}


def test_a_rejected_cell_carries_its_interval_and_badge():
    observations, ifs, wn2 = _board({"STRONG": [[-index / 12] * SLOTS for index in range(13)]})

    board = build_scoreboard(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    mean = _mean_cell(board)
    assert mean.badge == "beats IFS"
    assert mean.interval is not None
    assert mean.interval[1] < 0


def test_a_quarantined_cell_has_no_badge_or_interval():
    blocks = [
        [0.1 + 0.1 * (index + 1), 0.1 - 0.1 * (index + 1)] * (SLOTS // 2) for index in range(13)
    ]
    observations, ifs, wn2 = _board({TOWN: blocks})

    board = build_scoreboard(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    mean = _mean_cell(board)
    assert mean.quarantined
    assert mean.badge is None
    assert mean.interval is None


def test_written_document_round_trips(tmp_path: Path):
    observations, ifs, wn2 = _board(_constant_blocks([0.1] * 13))
    board = build_scoreboard(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])
    path = tmp_path / "scores" / "scores.json"

    write_scoreboard(board, path)

    assert Scoreboard.model_validate_json(path.read_bytes()) == board
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_local_files_are_read_in_the_scoreboard_units(tmp_path: Path):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "town,variable,valid_time,value\nKMIA,temperature_2m,2026-11-04T00:00:00+00:00,21.5\n",
        encoding="utf-8",
    )
    ifs = tmp_path / "ifs.csv"
    ifs.write_text(
        "town,variable,valid_time,lead_days,value\n"
        "KMIA,temperature_2m,2026-11-04T00:00:00+00:00,0,22.0\n",
        encoding="utf-8",
    )
    wn2 = tmp_path / "wn2"
    wn2.mkdir()
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
    ).to_parquet(wn2 / "2026-11-03-18.parquet", index=False)

    assert load_observations(observations)[0].value == 21.5
    assert load_ifs(ifs)[0].lead_days == 0
    assert load_wn2(wn2)[0].ens_mean == 21.0
    board = build_from_files(
        wn2=wn2,
        observations=observations,
        ifs=ifs,
        scoring_day=datetime(2026, 11, 5, tzinfo=UTC),
        lead_days=[0],
    )
    assert board.towns == ["KMIA"]
    assert len(board.cells) == 3
