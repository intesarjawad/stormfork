"""Read score inputs from local files and write the scoreboard document."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stormfork.scores.board import Scoreboard, build_scoreboard
from stormfork.scores.metrics import IfsForecast, Observation, Wn2Run

OBSERVATION_COLUMNS = ("town", "variable", "valid_time", "value")
IFS_COLUMNS = ("town", "variable", "valid_time", "lead_days", "value")
WN2_COLUMNS = ("town", "variable", "init_time", "valid_time", "ens_mean", "member_0")


def _utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        msg = f"{path} is missing columns: {sorted(missing)}"
        raise ValueError(msg)


def load_observations(path: Path) -> list[Observation]:
    """Read the station observations CSV written by the fetch command."""
    frame = pd.read_csv(path)
    _require_columns(frame, OBSERVATION_COLUMNS, path)
    return [
        Observation(
            town=str(row["town"]),
            variable=str(row["variable"]),
            valid_time=_utc(row["valid_time"]),
            value=float(row["value"]),
        )
        for row in frame.to_dict("records")
    ]


def load_ifs(path: Path) -> list[IfsForecast]:
    """Read the IFS forecast CSV written by the fetch command."""
    frame = pd.read_csv(path)
    _require_columns(frame, IFS_COLUMNS, path)
    return [
        IfsForecast(
            town=str(row["town"]),
            variable=str(row["variable"]),
            valid_time=_utc(row["valid_time"]),
            lead_days=int(row["lead_days"]),
            value=float(row["value"]),
        )
        for row in frame.to_dict("records")
    ]


def load_wn2(directory: Path) -> list[Wn2Run]:
    """Read every stored WN2 run in a directory of parquet files."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        msg = f"{directory} has no WN2 parquet files"
        raise ValueError(msg)
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    _require_columns(frame, WN2_COLUMNS, directory)
    return [
        Wn2Run(
            town=str(row["town"]),
            variable=str(row["variable"]),
            init_time=_utc(row["init_time"]),
            valid_time=_utc(row["valid_time"]),
            ens_mean=float(row["ens_mean"]),
            member_0=float(row["member_0"]),
        )
        for row in frame.to_dict("records")
    ]


def build_from_files(  # noqa: PLR0913 - one keyword per published input keeps the command explicit
    *,
    wn2: Path,
    observations: Path,
    ifs: Path,
    scoring_day: datetime,
    lead_days: Sequence[int],
    wn2_previous_runs_available: bool | None = None,
) -> Scoreboard:
    """Build the document from the three local inputs."""
    return build_scoreboard(
        load_observations(observations),
        load_ifs(ifs),
        load_wn2(wn2),
        scoring_day=scoring_day,
        lead_days=lead_days,
        wn2_previous_runs_available=wn2_previous_runs_available,
    )


def write_scoreboard(board: Scoreboard, path: Path) -> None:
    """Write the document with a trailing newline so a diff stays readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(board.model_dump_json(indent=2) + "\n", encoding="utf-8")
