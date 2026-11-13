import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from stormfork.scores.fetch import (
    PreviousRunsPayload,
    UnitMismatchError,
    parse_previous_runs,
    previous_runs_url,
    weighted_calls,
    wn2_url,
)
from stormfork.scores.inputs import ifs_url
from stormfork.scores.towns import load_towns

FIXTURES = Path("tests/fixtures")
TOWN = next(town for town in load_towns(Path("data/towns.json")) if town.icao == "KMIA")
START = datetime(2026, 11, 4, tzinfo=UTC)
END = datetime(2026, 11, 5, tzinfo=UTC)


def test_every_open_meteo_request_asks_for_metres_per_second():
    urls = [
        wn2_url(TOWN, START),
        ifs_url(TOWN, START, END),
        previous_runs_url(TOWN, START, END),
    ]

    for url in urls:
        assert parse_qs(urlsplit(url).query)["wind_speed_unit"] == ["ms"]


def test_pricing_page_examples():
    assert weighted_calls(15, 14) == 1.5
    assert weighted_calls(15, 28) == 3.0


def test_recorded_previous_runs_parses_wind_in_metres_per_second():
    body = (FIXTURES / "previous_runs.json").read_bytes()
    payload = PreviousRunsPayload.model_validate_json(body)
    key = "wind_speed_10m_previous_day1_ecmwf_ifs025"
    assert payload.hourly_units[key] == "m/s"

    rows = parse_previous_runs(body, TOWN)

    wind = [row for row in rows if row.model == "ecmwf_ifs025" and row.variable == "wind_speed_10m"]
    assert [(row.valid_time.strftime("%Y-%m-%dT%H:%M"), row.value) for row in wind] == [
        ("2026-11-04T00:00", 3.3),
        ("2026-11-04T06:00", 3.31),
        ("2026-11-04T12:00", 1.92),
        ("2026-11-04T18:00", 2.93),
        ("2026-11-05T00:00", 2.27),
        ("2026-11-05T06:00", 1.84),
        ("2026-11-05T12:00", 2.02),
        ("2026-11-05T18:00", 3.36),
    ]


def test_recorded_previous_runs_with_kilometres_per_hour_is_rejected():
    payload = json.loads((FIXTURES / "previous_runs.json").read_bytes())
    payload["hourly_units"]["wind_speed_10m_previous_day1_ecmwf_ifs025"] = "km/h"

    with pytest.raises(UnitMismatchError, match="wind_speed_10m_previous_day1_ecmwf_ifs025"):
        parse_previous_runs(json.dumps(payload).encode(), TOWN)
