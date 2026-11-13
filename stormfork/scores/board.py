"""Assemble the published scoreboard document from scored cells."""

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from stormfork.scores.metrics import (
    WINDOW_DAYS,
    Comparison,
    IfsForecast,
    Observation,
    Score,
    Wn2Run,
    comparisons,
    scores,
)
from stormfork.scores.multiplicity import AdjustedComparison, adjust

STALE_AFTER_DAYS = 2
UNITS = {"temperature_2m": "°C", "pressure_msl": "hPa", "wind_speed_10m": "m/s"}


class ScoreCell(BaseModel):
    """One model's score and test for a town, variable and lead."""

    model_config = ConfigDict(frozen=True)
    town: str
    variable: str
    lead_days: int
    model: str
    mae: float
    unit: str
    sample_size: int
    blocks: int
    statistic: float | None
    p_value: float | None
    mean_difference: float | None
    block_sd: float | None
    quarantined: bool
    rejected: bool
    interval: tuple[float, float] | None
    badge: str | None


class Scoreboard(BaseModel):
    """The document the site reads, with one cell per model, town, variable and lead."""

    model_config = ConfigDict(frozen=True)
    generated_at: date
    stale_since: date
    window_days: int
    lead_days: list[int]
    variables: list[str]
    models: list[str]
    towns: list[str]
    cells: list[ScoreCell]
    # Only the weekly probe sets this; other runs leave it unknown.
    wn2_previous_runs_available: bool | None = None


def _key(row: Score | Comparison) -> tuple[str, str, int, str]:
    return (row.model, row.variable, row.lead_days, row.town)


def _cell(score: Score, comparison: AdjustedComparison | None) -> ScoreCell:
    if comparison is None:
        return ScoreCell(
            model=score.model,
            variable=score.variable,
            lead_days=score.lead_days,
            town=score.town,
            mae=score.mae,
            unit=UNITS[score.variable],
            sample_size=score.sample_size,
            blocks=0,
            statistic=None,
            p_value=None,
            mean_difference=None,
            block_sd=None,
            quarantined=False,
            rejected=False,
            interval=None,
            badge=None,
        )
    row = comparison.comparison
    return ScoreCell(
        model=score.model,
        variable=score.variable,
        lead_days=score.lead_days,
        town=score.town,
        mae=score.mae,
        unit=UNITS[score.variable],
        sample_size=score.sample_size,
        blocks=row.blocks,
        statistic=row.statistic,
        p_value=row.p_value,
        mean_difference=row.mean_difference,
        block_sd=row.block_sd,
        quarantined=row.quarantined,
        rejected=comparison.rejected,
        interval=comparison.interval,
        badge=comparison.badge,
    )


def build_scoreboard(  # noqa: PLR0913 - each argument is one published input or option
    observations: Sequence[Observation],
    ifs: Sequence[IfsForecast],
    wn2: Sequence[Wn2Run],
    *,
    scoring_day: datetime,
    lead_days: Sequence[int],
    wn2_previous_runs_available: bool | None = None,
) -> Scoreboard:
    """Score every model and attach the adjusted test to each WN2 cell."""
    score_rows = scores(observations, ifs, wn2, scoring_day=scoring_day, lead_days=lead_days)
    adjusted = {
        _key(cell.comparison): cell
        for cell in adjust(
            comparisons(observations, ifs, wn2, scoring_day=scoring_day, lead_days=lead_days)
        )
    }
    cells = [_cell(row, adjusted.get(_key(row))) for row in score_rows]
    cells.sort(key=lambda cell: (cell.town, cell.variable, cell.lead_days, cell.model))
    towns = sorted(
        {row.town for row in observations} | {row.town for row in ifs} | {row.town for row in wn2}
    )
    return Scoreboard(
        generated_at=scoring_day.date(),
        stale_since=scoring_day.date() + timedelta(days=STALE_AFTER_DAYS),
        window_days=WINDOW_DAYS,
        lead_days=list(lead_days),
        variables=sorted({cell.variable for cell in cells}),
        models=sorted({cell.model for cell in cells}),
        towns=towns,
        cells=cells,
        wn2_previous_runs_available=wn2_previous_runs_available,
    )
