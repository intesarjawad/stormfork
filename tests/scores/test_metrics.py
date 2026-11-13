from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from stormfork.scores.metrics import (
    BLOCK_DAYS,
    IFS_MODEL,
    SLOTS_PER_BLOCK,
    STEP_HOURS,
    WN2_MEAN_MODEL,
    WN2_MEMBER_MODEL,
    Comparison,
    IfsForecast,
    Observation,
    Score,
    Wn2Run,
    comparisons,
    scores,
)

SCORING_DAY = datetime(2026, 11, 5, tzinfo=UTC)
WINDOW_START = SCORING_DAY - timedelta(days=91)
TOWN = "KMIA"
VARIABLE = "temperature_2m"


def _score(rows: Sequence[Score], model: str, lead: int) -> Score:
    return next(row for row in rows if row.model == model and row.lead_days == lead)


def _comparison(rows: Sequence[Comparison], model: str) -> Comparison:
    return next(row for row in rows if row.model == model)


def _block_series(
    block_means: Sequence[float], *, slots: Sequence[int] | None = None
) -> tuple[list[Observation], list[IfsForecast], list[Wn2Run]]:
    """Build 6-hourly observations, IFS forecasts and WN2 runs from block means.

    IFS predicts 1 and truth is 0, so a WN2 value of 1 + mean has the requested
    absolute-error differential.
    """
    observations: list[Observation] = []
    ifs: list[IfsForecast] = []
    wn2: list[Wn2Run] = []
    counts = slots or [SLOTS_PER_BLOCK // 2] * len(block_means)
    for block, (mean, count) in enumerate(zip(block_means, counts, strict=True)):
        for slot in range(count):
            valid_time = WINDOW_START + timedelta(days=BLOCK_DAYS * block, hours=STEP_HOURS * slot)
            observations.append(Observation(TOWN, VARIABLE, valid_time, 0.0))
            ifs.append(IfsForecast(TOWN, VARIABLE, valid_time, 0, 1.0))
            wn2.append(
                Wn2Run(
                    town=TOWN,
                    variable=VARIABLE,
                    init_time=valid_time,
                    valid_time=valid_time,
                    ens_mean=1.0 + mean,
                    member_0=1.0 + mean,
                )
            )
    return observations, ifs, wn2


def test_window_covers_91_days_and_excludes_the_scoring_day():
    observations = [
        Observation(TOWN, VARIABLE, WINDOW_START - timedelta(hours=6), 10.0),
        Observation(TOWN, VARIABLE, WINDOW_START, 0.0),
        Observation(TOWN, VARIABLE, SCORING_DAY - timedelta(hours=6), 2.0),
        Observation(TOWN, VARIABLE, SCORING_DAY, 10.0),
    ]
    ifs = [IfsForecast(row.town, row.variable, row.valid_time, 0, 0.0) for row in observations]

    rows = scores(observations, ifs, [], scoring_day=SCORING_DAY, lead_days=[0])

    assert [(row.mae, row.sample_size) for row in rows] == [(1.0, 2)]


def test_wn2_lead_requires_the_same_initialization_as_ifs():
    valid_time = WINDOW_START + timedelta(days=60)
    observations = [Observation(TOWN, VARIABLE, valid_time, 0.0)]
    ifs = [IfsForecast(TOWN, VARIABLE, valid_time, lead, 0.0) for lead in range(3)]
    wn2 = [
        Wn2Run(TOWN, VARIABLE, valid_time, valid_time, 1.0, 1.0),
        Wn2Run(TOWN, VARIABLE, valid_time - timedelta(hours=23), valid_time, 50.0, 50.0),
        Wn2Run(TOWN, VARIABLE, valid_time - timedelta(hours=24), valid_time, 10.0, 10.0),
        Wn2Run(TOWN, VARIABLE, valid_time - timedelta(hours=48), valid_time, 100.0, 100.0),
    ]

    rows = scores(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0, 1, 2])

    assert _score(rows, WN2_MEAN_MODEL, 0).mae == 1.0
    assert _score(rows, WN2_MEAN_MODEL, 1).mae == 10.0
    assert _score(rows, WN2_MEAN_MODEL, 2).mae == 100.0


def test_comparators_share_the_sample_when_wn2_is_shorter():
    times = [WINDOW_START + timedelta(hours=6 * index) for index in range(20)]
    observations = [Observation(TOWN, VARIABLE, time, 0.0) for time in times]
    ifs = [IfsForecast(TOWN, VARIABLE, time, 0, 1.0) for time in times]
    wn2 = [Wn2Run(TOWN, VARIABLE, time, time, 3.0, 3.0) for time in times[:10]]

    rows = scores(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    assert _score(rows, IFS_MODEL, 0).mae == 1.0
    assert _score(rows, IFS_MODEL, 0).sample_size == 10
    assert _score(rows, WN2_MEAN_MODEL, 0).mae == 3.0
    assert _score(rows, WN2_MEAN_MODEL, 0).sample_size == 10


def test_unmatched_ifs_error_cannot_reverse_the_model_comparison():
    times = [WINDOW_START, WINDOW_START + timedelta(days=1)]
    observations = [Observation(TOWN, VARIABLE, time, 0.0) for time in times]
    ifs = [
        IfsForecast(TOWN, VARIABLE, time, 0, error)
        for time, error in zip(times, [0.0, 100.0], strict=True)
    ]
    wn2 = [Wn2Run(TOWN, VARIABLE, times[0], times[0], 1.0, 1.0)]

    rows = scores(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    assert _score(rows, IFS_MODEL, 0).mae == 0
    assert _score(rows, WN2_MEAN_MODEL, 0).mae == 1
    assert {row.sample_size for row in rows} == {1}


def test_a_different_initialization_is_not_a_matched_lead():
    time = WINDOW_START + timedelta(days=2)
    observations = [Observation(TOWN, VARIABLE, time, 0.0)]
    ifs = [IfsForecast(TOWN, VARIABLE, time, 1, 0.0)]
    wn2 = [Wn2Run(TOWN, VARIABLE, time - timedelta(hours=30), time, 1.0, 1.0)]
    rows = scores(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[1])
    assert not any(row.model == WN2_MEAN_MODEL for row in rows)


def test_both_wn2_summaries_are_scored_separately():
    valid_time = WINDOW_START + timedelta(days=1)
    observations = [Observation(TOWN, VARIABLE, valid_time, 0.0)]
    ifs = [IfsForecast(TOWN, VARIABLE, valid_time, 0, 0.0)]
    wn2 = [Wn2Run(TOWN, VARIABLE, valid_time, valid_time, 1.0, 4.0)]

    rows = scores(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    assert _score(rows, WN2_MEAN_MODEL, 0).mae == 1.0
    assert _score(rows, WN2_MEMBER_MODEL, 0).mae == 4.0


def test_diebold_mariano_matches_hand_computed_batch_means():
    # Block means -k/12 give a mean differential of -0.5, a batch-means
    # statistic of -5.5549 on 12 degrees of freedom and a two-sided p of 1.25e-4.
    observations, ifs, wn2 = _block_series([-index / 12 for index in range(13)])

    rows = comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    row = _comparison(rows, WN2_MEAN_MODEL)
    assert row.blocks == 13
    assert row.statistic == pytest.approx(-5.5549, rel=1e-4)
    assert row.p_value == pytest.approx(1.2479e-4, rel=1e-3)


def test_a_block_with_fewer_than_half_its_slots_is_dropped():
    counts = [SLOTS_PER_BLOCK // 2] * 13
    counts[4] = SLOTS_PER_BLOCK // 2 - 1
    observations, ifs, wn2 = _block_series([0.05 * index for index in range(13)], slots=counts)

    rows = comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    row = _comparison(rows, WN2_MEAN_MODEL)
    assert row.blocks == 12
    assert row.p_value is not None


def test_fewer_than_eight_blocks_reports_not_enough_data():
    observations, ifs, wn2 = _block_series([0.05 * index for index in range(8)])
    tested = comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])
    assert _comparison(tested, WN2_MEAN_MODEL).p_value is not None

    observations, ifs, wn2 = _block_series([0.05 * index for index in range(7)])
    untested = comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    row = _comparison(untested, WN2_MEAN_MODEL)
    assert row.blocks == 7
    assert row.statistic is None
    assert row.p_value is None


def test_no_difference_across_equal_blocks_has_p_of_one():
    observations, ifs, wn2 = _block_series([0.0] * 13)

    rows = comparisons(observations, ifs, wn2, scoring_day=SCORING_DAY, lead_days=[0])

    row = _comparison(rows, WN2_MEAN_MODEL)
    assert row.statistic == 0.0
    assert row.p_value == 1.0
