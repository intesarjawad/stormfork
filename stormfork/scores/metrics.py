import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from scipy.stats import t as student_t

WINDOW_DAYS = 91
BLOCK_DAYS = 7
STEP_HOURS = 6
SLOTS_PER_BLOCK = BLOCK_DAYS * 24 // STEP_HOURS
MIN_BLOCKS = 8
ZERO_VARIANCE_REL_TOL = 1e-12
ZERO_MEAN_ABS_TOL = 1e-12
IFS_MODEL = "IFS"
WN2_MEAN_MODEL = "WN2 mean"
WN2_MEMBER_MODEL = "WN2 member 0"
WINDOW = timedelta(days=WINDOW_DAYS)
BLOCK = timedelta(days=BLOCK_DAYS)

logger = logging.getLogger(__name__)

Key = tuple[str, str, datetime]
Group = tuple[str, str]


@dataclass(frozen=True)
class Observation:
    """One measured value for a town, variable and valid time."""

    town: str
    variable: str
    valid_time: datetime
    value: float


@dataclass(frozen=True)
class IfsForecast:
    """One IFS forecast tied to the lead it was issued at."""

    town: str
    variable: str
    valid_time: datetime
    lead_days: int
    value: float


@dataclass(frozen=True)
class Wn2Run:
    """One stored WN2 run, retaining both published forecast summaries."""

    town: str
    variable: str
    init_time: datetime
    valid_time: datetime
    ens_mean: float
    member_0: float


@dataclass(frozen=True)
class Score:
    """Mean absolute error over a paired sample, in the variable's unit."""

    model: str
    variable: str
    lead_days: int
    town: str
    mae: float
    sample_size: int


@dataclass(frozen=True)
class Comparison:
    """Batch-means Diebold-Mariano test of one model's errors against IFS.

    A negative statistic favours the model. ``statistic`` and ``p_value``
    are null when fewer than eight seven-day blocks survive,
    which is the cell's "not enough data" state: it carries no badge or test.
    A cell whose block standard deviation is zero with a non-zero mean is
    quarantined: its identical offset in every block is a data fault, not
    evidence, so it also carries no test.
    """

    model: str
    variable: str
    lead_days: int
    town: str
    blocks: int
    statistic: float | None
    p_value: float | None
    mean_difference: float | None = None
    block_sd: float | None = None
    quarantined: bool = False


@dataclass(frozen=True)
class _BatchTest:
    """Test outcome before it is attached to a scoreboard cell."""

    mean_difference: float | None
    block_sd: float | None
    statistic: float | None
    p_value: float | None
    quarantined: bool


def scores(
    observations: Sequence[Observation],
    ifs: Sequence[IfsForecast],
    wn2: Sequence[Wn2Run],
    *,
    scoring_day: datetime,
    lead_days: Sequence[int],
) -> list[Score]:
    """Score displayed comparators on identical station, initialization and lead cells.

    IFS-only groups retain their observations. Where WN2 is present, all three
    displayed MAEs use its intersection with IFS and observations.
    """
    start = scoring_day - WINDOW
    truth = _truth(observations, start, scoring_day)
    rows: list[Score] = []
    for lead in lead_days:
        ifs_values = _ifs_values(ifs, lead, start, scoring_day)
        for model, values in _model_values(ifs_values, wn2, lead, start, scoring_day):
            paired = _paired_keys(model, values, truth, ifs_values)
            for group, errors in _grouped(values, paired, truth).items():
                rows.append(
                    Score(
                        model=model,
                        variable=group[1],
                        lead_days=lead,
                        town=group[0],
                        mae=fmean(errors),
                        sample_size=len(errors),
                    )
                )
    return sorted(rows, key=lambda row: (row.model, row.variable, row.lead_days, row.town))


def comparisons(
    observations: Sequence[Observation],
    ifs: Sequence[IfsForecast],
    wn2: Sequence[Wn2Run],
    *,
    scoring_day: datetime,
    lead_days: Sequence[int],
) -> list[Comparison]:
    """Test each WN2 summary against IFS on the window's paired samples.

    The loss differential is the model's absolute error minus IFS's, so a
    negative mean favours WN2. Thirteen weekly batch means feed a Student t
    reference: 91 days of storm-driven errors have heavy tails, which makes a
    normal reference too liberal.
    """
    start = scoring_day - WINDOW
    truth = _truth(observations, start, scoring_day)
    rows: list[Comparison] = []
    for lead in lead_days:
        ifs_values = _ifs_values(ifs, lead, start, scoring_day)
        for model, values in _model_values(ifs_values, wn2, lead, start, scoring_day)[1:]:
            paired = _paired_keys(model, values, truth, ifs_values)
            for group, differences in _differences(values, ifs_values, truth, paired).items():
                block_means = _block_means(differences, start)
                test = _diebold_mariano(block_means)
                if test.quarantined:
                    logger.warning(
                        "quarantined degenerate cell: model=%s variable=%s lead_days=%d town=%s",
                        model,
                        group[1],
                        lead,
                        group[0],
                    )
                rows.append(
                    Comparison(
                        model=model,
                        variable=group[1],
                        lead_days=lead,
                        town=group[0],
                        blocks=len(block_means),
                        statistic=test.statistic,
                        p_value=test.p_value,
                        mean_difference=test.mean_difference,
                        block_sd=test.block_sd,
                        quarantined=test.quarantined,
                    )
                )
    return sorted(rows, key=lambda row: (row.model, row.variable, row.lead_days, row.town))


def _truth(observations: Sequence[Observation], start: datetime, end: datetime) -> dict[Key, float]:
    return {
        (row.town, row.variable, row.valid_time): row.value
        for row in observations
        if start <= row.valid_time < end
    }


def _ifs_values(
    ifs: Sequence[IfsForecast], lead: int, start: datetime, end: datetime
) -> dict[Key, float]:
    return {
        (row.town, row.variable, row.valid_time): row.value
        for row in ifs
        if row.lead_days == lead and start <= row.valid_time < end
    }


def _model_values(
    ifs_values: dict[Key, float],
    wn2: Sequence[Wn2Run],
    lead: int,
    start: datetime,
    end: datetime,
) -> list[tuple[str, dict[Key, float]]]:
    mean, member = _select_runs(wn2, lead, start, end)
    compared_groups = {(key[0], key[1]) for key in mean}
    ifs_values = {
        key: value
        for key, value in ifs_values.items()
        if (key[0], key[1]) not in compared_groups or key in mean
    }
    return [
        (IFS_MODEL, ifs_values),
        (WN2_MEAN_MODEL, mean),
        (WN2_MEMBER_MODEL, member),
    ]


def _select_runs(
    wn2: Sequence[Wn2Run], lead: int, start: datetime, end: datetime
) -> tuple[dict[Key, float], dict[Key, float]]:
    chosen: dict[Key, Wn2Run] = {}
    for run in wn2:
        if not start <= run.valid_time < end:
            continue
        if run.init_time != run.valid_time - timedelta(days=lead):
            continue
        key = (run.town, run.variable, run.valid_time)
        incumbent = chosen.get(key)
        if incumbent is None or run.init_time > incumbent.init_time:
            chosen[key] = run
    return (
        {key: run.ens_mean for key, run in chosen.items()},
        {key: run.member_0 for key, run in chosen.items()},
    )


def _paired_keys(
    model: str, values: dict[Key, float], truth: dict[Key, float], ifs_values: dict[Key, float]
) -> set[Key]:
    paired = values.keys() & truth.keys()
    return paired if model == IFS_MODEL else paired & ifs_values.keys()


def _grouped(
    values: dict[Key, float], paired: set[Key], truth: dict[Key, float]
) -> dict[Group, list[float]]:
    groups: dict[Group, list[float]] = {}
    for key in paired:
        groups.setdefault((key[0], key[1]), []).append(abs(values[key] - truth[key]))
    return groups


def _differences(
    values: dict[Key, float],
    ifs_values: dict[Key, float],
    truth: dict[Key, float],
    paired: set[Key],
) -> dict[Group, list[tuple[datetime, float]]]:
    groups: dict[Group, list[tuple[datetime, float]]] = {}
    for key in paired:
        difference = abs(values[key] - truth[key]) - abs(ifs_values[key] - truth[key])
        groups.setdefault((key[0], key[1]), []).append((key[2], difference))
    return groups


def _block_means(differences: Sequence[tuple[datetime, float]], start: datetime) -> list[float]:
    blocks: dict[int, list[float]] = {}
    for valid_time, difference in differences:
        blocks.setdefault((valid_time - start) // BLOCK, []).append(difference)
    return [fmean(values) for values in blocks.values() if len(values) * 2 >= SLOTS_PER_BLOCK]


def _diebold_mariano(block_means: Sequence[float]) -> _BatchTest:
    count = len(block_means)
    if count == 0:
        return _BatchTest(None, None, None, None, quarantined=False)
    mean = fmean(block_means)
    if count < MIN_BLOCKS:
        block_sd = _sample_sd(block_means, mean) if count > 1 else None
        return _BatchTest(mean, block_sd, None, None, quarantined=False)
    variance = sum((value - mean) ** 2 for value in block_means) / (count - 1)
    block_sd = math.sqrt(variance)
    if block_sd <= ZERO_VARIANCE_REL_TOL * max(1.0, abs(mean)):
        if abs(mean) <= ZERO_MEAN_ABS_TOL:
            return _BatchTest(0.0, 0.0, 0.0, 1.0, quarantined=False)
        return _BatchTest(mean, block_sd, None, None, quarantined=True)
    statistic = mean / (block_sd / math.sqrt(count))
    return _BatchTest(
        mean,
        block_sd,
        statistic,
        2 * float(student_t.sf(abs(statistic), df=count - 1)),
        quarantined=False,
    )


def _sample_sd(values: Sequence[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
