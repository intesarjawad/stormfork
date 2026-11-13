import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import t as student_t

from stormfork.scores.metrics import Comparison
from stormfork.scores.multiplicity import (
    BEATS_IFS,
    AdjustedComparison,
    adjust,
    adjusted_interval,
    benjamini_hochberg,
)

P_VALUES = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
BLOCKS = 13
SLOTS = 14


def _cell(
    p_value: float, *, blocks: int = BLOCKS, block_sd: float = 1.0, direction: int = 1
) -> Comparison:
    """Build a cell whose statistic is consistent with its p-value."""
    critical = 0.0 if p_value == 1 else float(student_t.isf(p_value / 2, df=blocks - 1))
    mean_difference = direction * critical * block_sd / math.sqrt(blocks)
    return Comparison(
        model="WN2 mean",
        variable="temperature_2m",
        lead_days=0,
        town="KMIA",
        blocks=blocks,
        statistic=direction * critical,
        p_value=p_value,
        mean_difference=mean_difference,
        block_sd=block_sd,
    )


def _excludes_zero(cell: AdjustedComparison) -> bool:
    """Treat an endpoint touching zero as excluding it at the BH boundary."""
    interval = cell.interval
    if interval is None or cell.comparison.mean_difference is None:
        return False
    if cell.comparison.mean_difference < 0:
        return interval[1] <= 0
    if cell.comparison.mean_difference > 0:
        return interval[0] >= 0
    return False


def _ar1_errors(rng: np.random.Generator, cells: int, steps: int) -> np.ndarray:
    """Student-t 3 innovations through a six-hour AR(1) with phi 0.6."""
    innovations = rng.standard_t(3, size=(cells, steps))
    errors = np.empty_like(innovations)
    errors[:, 0] = innovations[:, 0]
    for step in range(1, steps):
        errors[:, step] = 0.6 * errors[:, step - 1] + innovations[:, step]
    return errors


def _null_p_values(seed: int) -> list[float]:
    block_means = (
        _ar1_errors(np.random.default_rng(seed), 1000, BLOCKS * SLOTS)
        .reshape(1000, BLOCKS, SLOTS)
        .mean(axis=2)
    )
    differences = block_means.mean(axis=1)
    block_sd = block_means.std(axis=1, ddof=1)
    statistics = differences / (block_sd / math.sqrt(BLOCKS))
    p_values = 2 * student_t.sf(np.abs(statistics), df=BLOCKS - 1)
    return [float(value) for value in p_values]


def test_benjamini_hochberg_rejects_exactly_the_first_two():
    rejected = benjamini_hochberg(P_VALUES)

    assert rejected == [True, True, *[False] * 8]
    assert sum(rejected) == 2


def test_adjusted_intervals_exclude_zero_for_exactly_the_rejected_cells():
    rows = [
        _cell(p_value, direction=1 if index % 2 else -1) for index, p_value in enumerate(P_VALUES)
    ]

    adjusted = adjust(rows)

    assert [cell.rejected for cell in adjusted] == benjamini_hochberg(P_VALUES)
    for cell in adjusted:
        assert cell.interval is not None
        assert _excludes_zero(cell) == cell.rejected


@given(
    st.lists(
        st.floats(min_value=1e-12, max_value=1 - 1e-12, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=25,
    ),
    st.floats(min_value=0.05, max_value=5, allow_nan=False, allow_infinity=False),
    st.integers(min_value=8, max_value=20),
)
def test_badges_match_intervals_excluding_zero(p_values: list[float], block_sd: float, blocks: int):
    rows = [
        _cell(p_value, blocks=blocks, block_sd=block_sd, direction=1 if index % 2 else -1)
        for index, p_value in enumerate(p_values)
    ]

    adjusted = adjust(rows)

    assert [cell.rejected for cell in adjusted] == benjamini_hochberg(p_values)
    for cell in adjusted:
        assert cell.interval is not None
        # A p-value within rounding of its BH threshold puts an endpoint on zero, where the
        # interval cannot say which side of the decision the cell fell.
        if min(abs(cell.interval[0]), abs(cell.interval[1])) <= 1e-9 * block_sd:
            continue
        assert (cell.badge is not None) == _excludes_zero(cell)


def test_all_null_boards_rarely_produce_any_badge():
    seeds_with_a_badge = sum(any(benjamini_hochberg(_null_p_values(seed))) for seed in range(100))

    assert seeds_with_a_badge <= 10


def test_unadjusted_interval_covers_the_true_mean_in_seeded_trials():
    true_mean = 0.5
    covered = 0
    for seed in range(100):
        block_means = (
            _ar1_errors(np.random.default_rng(seed + 500), 1, BLOCKS * SLOTS)
            .reshape(BLOCKS, SLOTS)
            .mean(axis=1)
            + true_mean
        )
        mean_difference = float(block_means.mean())
        block_sd = float(block_means.std(ddof=1))
        low, high = adjusted_interval(mean_difference, block_sd, BLOCKS, alpha=0.05)
        covered += low <= true_mean <= high

    assert covered >= 93


def test_a_directional_badge_follows_the_sign_of_a_rejected_cell():
    adjusted = adjust([_cell(0.001, direction=-1), _cell(0.001, direction=1)])
    assert [cell.badge for cell in adjusted] == [BEATS_IFS, "IFS better"]


@pytest.mark.parametrize("p_value", [-0.1, 1.1, math.nan, math.inf])
def test_benjamini_hochberg_rejects_invalid_probabilities(p_value: float):
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([p_value])
