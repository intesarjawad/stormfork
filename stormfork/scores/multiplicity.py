"""Benjamini-Hochberg control for the scoreboard's many forecast cells."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import t as student_t

from stormfork.scores.metrics import Comparison

Q = 0.05
MIN_INTERVAL_BLOCKS = 2
BEATS_IFS = "beats IFS"
IFS_BETTER = "IFS better"


@dataclass(frozen=True)
class AdjustedComparison:
    """One cell's test after false-discovery-rate adjustment.

    A cell without a test carries no rejection, interval or badge.
    """

    comparison: Comparison
    rejected: bool
    interval: tuple[float, float] | None
    badge: str | None


def benjamini_hochberg(p_values: Sequence[float], q: float = Q) -> list[bool]:
    """Reject every hypothesis in the largest Benjamini-Hochberg step-up set.

    Raises:
        ValueError: A supplied p-value is nonfinite or outside [0, 1],
            or q is outside (0, 1).
    """
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        msg = "p-values must be finite and in [0, 1]"
        raise ValueError(msg)
    if not 0 < q < 1:
        msg = "q must be in (0, 1)"
        raise ValueError(msg)
    count = len(p_values)
    if count == 0:
        return []
    cutoff: float | None = None
    for rank, p_value in enumerate(sorted(p_values), start=1):
        if p_value <= rank * q / count:
            cutoff = rank * q / count
    return [cutoff is not None and p_value <= cutoff for p_value in p_values]


def false_coverage_alpha(tests: int, rejections: int, q: float = Q) -> float:
    """Return the false-coverage-rate alpha matched to a BH rejection count.

    Raises:
        ValueError: There are no tests or q is outside (0, 1).
    """
    if tests <= 0:
        msg = "tests must be positive"
        raise ValueError(msg)
    if not 0 < q < 1:
        msg = "q must be in (0, 1)"
        raise ValueError(msg)
    return max(rejections, 1) * q / tests


def adjusted_interval(
    mean_difference: float,
    block_sd: float,
    blocks: int,
    alpha: float,
) -> tuple[float, float]:
    """Return the false-coverage-rate interval matched to a BH rejection.

    Benjamini and Yekutieli (2005) replace the usual 1-alpha coverage with
    alpha = max(R, 1) q / m, which makes an interval exclude zero exactly when
    its cell is rejected.

    Raises:
        ValueError: Alpha is outside (0, 1) or there are too few blocks for a
            Student t reference.
    """
    if not 0 < alpha < 1:
        msg = "alpha must be in (0, 1)"
        raise ValueError(msg)
    if blocks < MIN_INTERVAL_BLOCKS:
        msg = "blocks must be at least two for a Student t reference"
        raise ValueError(msg)
    margin = float(student_t.isf(alpha / 2, df=blocks - 1)) * block_sd / math.sqrt(blocks)
    return (mean_difference - margin, mean_difference + margin)


def badge(*, rejected: bool, mean_difference: float) -> str | None:
    """Name the direction only for a rejected cell."""
    if not rejected or mean_difference == 0:
        return None
    return BEATS_IFS if mean_difference < 0 else IFS_BETTER


def tested_count(comparisons: Sequence[Comparison]) -> int:
    """Count the cells that enter the family: tested and not quarantined."""
    return sum(row.p_value is not None and not row.quarantined for row in comparisons)


def adjust(comparisons: Sequence[Comparison], q: float = Q) -> list[AdjustedComparison]:
    """Apply one BH correction and matching intervals across the whole board."""
    tested = [row for row in comparisons if row.p_value is not None and not row.quarantined]
    p_values = [row.p_value for row in tested if row.p_value is not None]
    rejected = benjamini_hochberg(p_values, q)
    count = len(p_values)
    rejections = sum(rejected)
    alpha = false_coverage_alpha(count, rejections, q) if count else 0.0
    cells: list[AdjustedComparison] = []
    position = 0
    for row in comparisons:
        if row.p_value is None or row.quarantined:
            cells.append(AdjustedComparison(row, rejected=False, interval=None, badge=None))
            continue
        is_rejected = rejected[position]
        position += 1
        if row.mean_difference is None or row.block_sd is None:
            cells.append(
                AdjustedComparison(comparison=row, rejected=is_rejected, interval=None, badge=None)
            )
            continue
        interval = adjusted_interval(row.mean_difference, row.block_sd, row.blocks, alpha)
        cells.append(
            AdjustedComparison(
                comparison=row,
                rejected=is_rejected,
                interval=interval,
                badge=badge(rejected=is_rejected, mean_difference=row.mean_difference),
            )
        )
    return cells
