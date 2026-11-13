import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from stormfork.scores.metrics import (
    BLOCK_DAYS,
    SLOTS_PER_BLOCK,
    STEP_HOURS,
    WN2_MEAN_MODEL,
    ZERO_VARIANCE_REL_TOL,
    Comparison,
    IfsForecast,
    Observation,
    Wn2Run,
    comparisons,
)
from stormfork.scores.multiplicity import adjust
from stormfork.scores.multiplicity import tested_count as count_tested

SCORING_DAY = datetime(2026, 11, 5, tzinfo=UTC)
WINDOW_START = SCORING_DAY - timedelta(days=91)
TOWN = "KMIA"
VARIABLE = "temperature_2m"
SLOTS = SLOTS_PER_BLOCK // 2
FIXED_MEANS = [0.3, -0.1, 0.2, 0.0, 0.4, -0.2, 0.1, 0.3, -0.1, 0.2, 0.0, 0.1, 0.1]


def _board(
    series: Mapping[str, Sequence[Sequence[float]]], *, ifs_error: float = 10.0
) -> tuple[list[Observation], list[IfsForecast], list[Wn2Run]]:
    """Build paired six-hourly samples whose loss differential follows each block."""
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


def _mean_rows(rows: Sequence[Comparison]) -> list[Comparison]:
    return [row for row in rows if row.model == WN2_MEAN_MODEL]


def _by_town(rows: Sequence[Comparison], town: str) -> Comparison:
    return next(row for row in rows if row.town == town)


def test_fixed_board_matches_hand_computed_batch_means():
    series = {"T000": [[mean] * SLOTS for mean in FIXED_MEANS]}
    for index in range(1, 1000):
        series[f"T{index:03d}"] = [[0.0] * SLOTS for _ in range(13)]
    observations, ifs, wn2 = _board(series)

    rows = _mean_rows(comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0]))

    row = _by_town(rows, "T000")
    assert row.blocks == 13
    assert row.mean_difference == pytest.approx(0.1, abs=1e-4)
    assert row.statistic == pytest.approx(2.0261, abs=1e-4)
    assert row.p_value == pytest.approx(0.06557, abs=1e-4)
    assert count_tested(rows) == 1000
    adjusted = adjust(rows)
    special = next(cell for cell in adjusted if cell.comparison.town == "T000")
    assert not special.rejected
    assert special.badge is None
    assert special.interval == pytest.approx((-0.2032, 0.4032), abs=1e-4)
    assert all(cell.badge is None for cell in adjusted)


def test_identical_series_has_zero_difference_and_an_empty_interval():
    observations, ifs, wn2 = _board(_constant_blocks([0.0] * 13))

    rows = _mean_rows(comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0]))

    row = _by_town(rows, TOWN)
    assert row.mean_difference == 0.0
    assert row.block_sd == 0.0
    assert row.statistic == 0.0
    assert row.p_value == 1.0
    assert not row.quarantined
    assert count_tested(rows) == 1
    cell = adjust(rows)[0]
    assert cell.interval == (0.0, 0.0)
    assert cell.badge is None


def test_float_rounded_constant_offset_is_quarantined(
    caplog: pytest.LogCaptureFixture,
):
    blocks = [
        [0.1 + 0.1 * (index + 1), 0.1 - 0.1 * (index + 1)] * (SLOTS // 2) for index in range(13)
    ]
    observations, ifs, wn2 = _board(
        {
            TOWN: blocks,
            "STRONG": [[-index / 12] * SLOTS for index in range(13)],
            "FLAT": [[0.0] * SLOTS for _ in range(13)],
        }
    )

    with caplog.at_level(logging.WARNING, logger="stormfork.scores.metrics"):
        rows = _mean_rows(
            comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])
        )

    row = _by_town(rows, TOWN)
    assert row.quarantined
    assert row.statistic is None
    assert row.p_value is None
    assert row.mean_difference == pytest.approx(0.1)
    assert row.block_sd is not None
    assert row.block_sd <= ZERO_VARIANCE_REL_TOL * max(1.0, abs(row.mean_difference or 0.0))
    assert TOWN in caplog.text
    assert count_tested(rows) == 2

    adjusted = adjust(rows)
    quarantined = next(cell for cell in adjusted if cell.comparison.town == TOWN)
    assert quarantined.interval is None
    assert quarantined.badge is None
    assert not quarantined.rejected

    kept = [cell for cell in adjusted if cell.comparison.town != TOWN]
    without_fault = adjust([row for row in rows if row.town != TOWN])
    assert [(cell.rejected, cell.interval, cell.badge) for cell in kept] == [
        (cell.rejected, cell.interval, cell.badge) for cell in without_fault
    ]
    strong = next(cell for cell in kept if cell.comparison.town == "STRONG")
    assert strong.badge == "beats IFS"
    assert strong.interval is not None
    assert strong.interval[1] < 0


def test_a_cell_with_seven_blocks_is_not_tested():
    observations, ifs, wn2 = _board(_constant_blocks([0.1] * 7))

    rows = _mean_rows(comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0]))

    row = _by_town(rows, TOWN)
    assert row.blocks == 7
    assert row.statistic is None
    assert row.p_value is None
    assert not row.quarantined
    assert count_tested(rows) == 0
    cell = adjust(rows)[0]
    assert cell.interval is None
    assert cell.badge is None


def test_a_board_with_no_tested_cell_has_no_intervals_or_badges():
    observations, ifs, wn2 = _board(
        {
            "AAA": [[0.1] * SLOTS for _ in range(7)],
            "BBB": [[0.2] * SLOTS for _ in range(7)],
        }
    )

    rows = _mean_rows(comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0]))

    assert count_tested(rows) == 0
    adjusted = adjust(rows)
    assert all(cell.interval is None and cell.badge is None for cell in adjusted)
