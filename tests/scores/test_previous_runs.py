import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest

from stormfork.errors import UserError
from stormfork.scores.fetch import (
    DAILY_CALL_LIMIT,
    PREVIOUS_RUN_MODELS,
    SCORING_HOURS,
    backfill_plan,
    collect_backfill,
    nearest_observations,
    parse_previous_runs,
    probe_wn2_previous_runs,
)
from stormfork.scores.metrics import Observation
from stormfork.scores.towns import load_towns
from tests.test_cli import invoke

FIXTURES = Path("tests/fixtures")
TOWNS = load_towns(Path("data/towns.json"))
TOWN = next(town for town in TOWNS if town.icao == "KMIA")
START = datetime(2026, 11, 4, tzinfo=UTC)
UNTIL = datetime(2026, 11, 6, tzinfo=UTC)


def _recorded_request(url: str) -> bytes:
    if urlparse(url).hostname == "previous-runs-api.open-meteo.com":
        return (FIXTURES / "previous_runs.json").read_bytes()
    return (FIXTURES / "asos_chunk.csv").read_bytes()


def test_recorded_previous_runs_keep_only_the_six_hourly_grid():
    rows = parse_previous_runs((FIXTURES / "previous_runs.json").read_bytes(), TOWN)

    assert {row.valid_time.hour for row in rows} == set(SCORING_HOURS)
    assert {row.lead_days for row in rows} == {1}
    assert {row.town for row in rows} == {"KMIA"}
    assert {row.model for row in rows} == set(PREVIOUS_RUN_MODELS)
    series = {(row.model, row.variable) for row in rows}
    assert len(series) == 27
    assert ("gfs025", "wind_speed_10m") not in series
    assert ("gfs025", "pressure_msl") in series
    assert len(rows) == 27 * 8


def test_nearest_report_within_thirty_minutes_is_selected():
    observations = [
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 3, 23, 53, tzinfo=UTC), 1.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 0, 53, tzinfo=UTC), 2.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 5, 53, tzinfo=UTC), 3.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 6, 53, tzinfo=UTC), 4.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 11, 53, tzinfo=UTC), 5.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 12, 53, tzinfo=UTC), 6.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 17, 53, tzinfo=UTC), 7.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 18, 53, tzinfo=UTC), 8.0),
    ]
    targets = [
        datetime(2026, 11, 4, 0, tzinfo=UTC),
        datetime(2026, 11, 4, 6, tzinfo=UTC),
        datetime(2026, 11, 4, 12, tzinfo=UTC),
        datetime(2026, 11, 4, 18, tzinfo=UTC),
    ]

    aligned = nearest_observations(observations, targets)

    assert [(row.valid_time.hour, row.value) for row in aligned] == [
        (0, 1.0),
        (6, 3.0),
        (12, 5.0),
        (18, 7.0),
    ]


def test_a_report_outside_thirty_minutes_is_left_out():
    observations = [
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 0, 53, tzinfo=UTC), 2.0)
    ]
    target = datetime(2026, 11, 4, 3, tzinfo=UTC)

    assert nearest_observations(observations, [target]) == []


def test_a_tie_uses_the_earlier_report():
    observations = [
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 5, 45, tzinfo=UTC), 1.0),
        Observation("KMIA", "temperature_2m", datetime(2026, 11, 4, 6, 15, tzinfo=UTC), 2.0),
    ]

    aligned = nearest_observations(observations, [datetime(2026, 11, 4, 6, tzinfo=UTC)])

    assert [row.value for row in aligned] == [1.0]


def test_backfill_plan_stores_ninety_day_chunks_under_the_daily_limit():
    days = backfill_plan(
        TOWNS,
        since=datetime(2024, 1, 1, tzinfo=UTC),
        until=datetime(2024, 4, 1, tzinfo=UTC),
    )

    assert [(day.start.isoformat(), day.end.isoformat()) for day in days] == [
        ("2024-01-01", "2024-03-30"),
        ("2024-03-31", "2024-03-31"),
    ]
    assert days[0].calls == pytest.approx(555.43, abs=0.01)
    assert all(day.calls < DAILY_CALL_LIMIT for day in days)


def test_dry_run_prints_the_weighted_calls_of_every_planned_day(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    invoke(
        monkeypatch,
        "scores",
        "build",
        "--since",
        "2024-01-01",
        "--until",
        "2024-04-01",
        "--dry-run",
    )

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0] == "2024-01-01 to 2024-03-30: 555.4 weighted calls"
    for line in lines:
        assert float(line.split(":")[1].split()[0]) < DAILY_CALL_LIMIT


def test_collector_writes_one_chunk_and_resumes_past_it(tmp_path: Path):
    day = collect_backfill([TOWN], tmp_path, since=START, until=UNTIL, request=_recorded_request)

    assert day is not None
    assert day.start.isoformat() == "2026-11-04"
    runs = pd.read_parquet(tmp_path / "previous_runs/2026-11-04.parquet")
    observations = pd.read_parquet(tmp_path / "observations/2026-11-04.parquet")
    assert list(runs.columns) == [
        "town",
        "model",
        "variable",
        "valid_time",
        "lead_days",
        "value",
    ]
    assert list(observations.columns) == ["town", "variable", "valid_time", "value"]
    assert set(runs["town"]) == {"KMIA"}
    assert set(observations["town"]) == {"KMIA"}
    assert {row.hour for row in runs["valid_time"]} == set(SCORING_HOURS)
    assert {row.hour for row in observations["valid_time"]} == set(SCORING_HOURS)
    first = observations[
        (observations["variable"] == "temperature_2m")
        & (observations["valid_time"] == pd.Timestamp("2026-11-04T00:00Z"))
    ]
    assert first["value"].iloc[0] == pytest.approx((78.0 - 32.0) * 5 / 9)
    at_six = observations[
        (observations["variable"] == "wind_speed_10m")
        & (observations["valid_time"] == pd.Timestamp("2026-11-04T06:00Z"))
    ]
    assert at_six["value"].iloc[0] == pytest.approx(3.0 * 0.514444)
    assert not list(tmp_path.rglob("*.tmp"))

    assert (
        collect_backfill([TOWN], tmp_path, since=START, until=UNTIL, request=_recorded_request)
        is None
    )


def test_a_chunk_over_the_daily_limit_is_refused(tmp_path: Path):
    with pytest.raises(UserError, match="over the 1 limit") as raised:
        collect_backfill(
            [TOWN], tmp_path, since=START, until=UNTIL, request=_recorded_request, limit=1.0
        )

    assert raised.value.exit_code == 3
    assert not list(tmp_path.iterdir())


def test_probe_reports_the_recorded_null_archive():
    body = (FIXTURES / "wn2_previous_runs.json").read_bytes()

    available = probe_wn2_previous_runs(
        TOWN, now=datetime(2026, 11, 7, tzinfo=UTC), request=lambda _url: body
    )

    assert available is False


def test_probe_reports_a_populated_archive():
    payload = json.loads((FIXTURES / "wn2_previous_runs.json").read_bytes())
    payload["hourly"]["temperature_2m_previous_day1"][0] = 24.5

    available = probe_wn2_previous_runs(
        TOWN,
        now=datetime(2026, 11, 7, tzinfo=UTC),
        request=lambda _url: json.dumps(payload).encode(),
    )

    assert available is True


def test_probe_window_covers_the_last_seven_days():
    urls: list[str] = []

    def recorded(url: str) -> bytes:
        urls.append(url)
        return (FIXTURES / "wn2_previous_runs.json").read_bytes()

    probe_wn2_previous_runs(TOWN, now=datetime(2026, 11, 7, tzinfo=UTC), request=recorded)

    assert "start_date=2026-10-31" in urls[0]
    assert "end_date=2026-11-07" in urls[0]
    assert "models=google_weathernext2_ensemble" in urls[0]
