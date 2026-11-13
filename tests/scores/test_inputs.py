import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from stormfork.scores.build import load_ifs, load_observations
from stormfork.scores.inputs import fetch_inputs, parse_asos, parse_ifs
from stormfork.scores.metrics import scores
from stormfork.scores.towns import load_towns

TOWN = next(town for town in load_towns(Path("data/towns.json")) if town.icao == "KMIA")
ASOS = b"""station,valid,tmpf,mslp,sknt
KMIA,2026-11-05 00:53,84.20,1015.2,5.00
KMIA,2026-11-05 01:53,M,M,M
"""
IFS_UNITS: dict[str, str] = {
    "time": "iso8601",
    "temperature_2m": "°C",
    "pressure_msl": "hPa",
    "wind_speed_10m": "m/s",
}
IFS: dict[str, object] = {
    "utc_offset_seconds": 0,
    "hourly_units": IFS_UNITS,
    "hourly": {
        "time": ["2026-11-05T00:00"],
        "temperature_2m": [22.0],
        "pressure_msl": [1012.0],
        "wind_speed_10m": [3.0],
    },
}


def test_asos_rows_are_converted_to_scoreboard_units():
    rows = parse_asos(ASOS, TOWN)

    assert len(rows) == 3
    temperature = next(row for row in rows if row.variable == "temperature_2m")
    pressure = next(row for row in rows if row.variable == "pressure_msl")
    wind = next(row for row in rows if row.variable == "wind_speed_10m")
    assert temperature.value == pytest.approx(29.0)
    assert pressure.value == pytest.approx(1015.2)
    assert wind.value == pytest.approx(2.57222)
    assert temperature.valid_time.isoformat() == "2026-11-05T00:53:00+00:00"


def test_ifs_rows_are_lead_zero_comparisons():
    rows = parse_ifs(json.dumps(IFS).encode(), TOWN)

    assert len(rows) == 3
    assert {row.variable for row in rows} == {
        "temperature_2m",
        "pressure_msl",
        "wind_speed_10m",
    }
    assert all(row.lead_days == 0 for row in rows)
    assert all(row.town == "KMIA" for row in rows)


def test_ifs_rejects_a_non_utc_response():
    payload = {**IFS, "utc_offset_seconds": 3600}

    with pytest.raises(ValueError, match="UTC"):
        parse_ifs(json.dumps(payload).encode(), TOWN)


def test_ifs_rejects_wrong_units():
    payload = {**IFS, "hourly_units": {**IFS_UNITS, "temperature_2m": "K"}}

    with pytest.raises(ValueError, match="temperature_2m"):
        parse_ifs(json.dumps(payload).encode(), TOWN)


def test_fetch_aligns_reports_across_the_start_boundary(tmp_path: Path):
    start = datetime(2026, 11, 5, tzinfo=UTC)

    def request(url: str) -> bytes:
        if "asos.py" in url:
            query = parse_qs(urlsplit(url).query)
            assert query["day1"] == ["4"]
            return b"station,valid,tmpf,mslp,sknt\nKMIA,2026-11-04 23:53,68,1015,10\n"
        return json.dumps(IFS).encode()

    observations_path, ifs_path = fetch_inputs(
        [TOWN], start=start, end=start + timedelta(days=1), output=tmp_path, request=request
    )
    observations = load_observations(observations_path)
    assert len(observations) == 3
    assert {row.valid_time for row in observations} == {start}
    assert (
        len(
            scores(
                observations,
                load_ifs(ifs_path),
                [],
                scoring_day=start + timedelta(days=1),
                lead_days=[0],
            )
        )
        == 3
    )


@pytest.mark.parametrize(("minute", "count"), [(30, 3), (29, 0)])
def test_fetch_keeps_only_reports_within_thirty_minutes(tmp_path: Path, minute: int, count: int):
    start = datetime(2026, 11, 5, tzinfo=UTC)

    def request(url: str) -> bytes:
        if "asos.py" in url:
            return (
                f"station,valid,tmpf,mslp,sknt\nKMIA,2026-11-04 23:{minute},68,1015,10\n".encode()
            )
        return json.dumps(IFS).encode()

    path, _ = fetch_inputs(
        [TOWN], start=start, end=start + timedelta(days=1), output=tmp_path, request=request
    )
    assert len(load_observations(path)) == count


def test_complete_input_window_records_the_scoring_day(tmp_path: Path):
    start = datetime(2026, 11, 4, tzinfo=UTC)
    end = datetime(2026, 11, 5, tzinfo=UTC)
    fetch_inputs(
        [TOWN],
        start=start,
        end=end,
        output=tmp_path,
        request=lambda url: ASOS if "asos.py" in url else json.dumps(IFS).encode(),
    )
    assert (tmp_path / "scoring-day.txt").read_text() == "2026-11-05\n"


def test_failed_refresh_keeps_the_complete_stored_input_window(tmp_path: Path):
    stored = {
        "observations.csv": b"stored observations",
        "ifs.csv": b"stored forecasts",
        "scoring-day.txt": b"2026-11-05\n",
    }
    for name, data in stored.items():
        (tmp_path / name).write_bytes(data)

    def unavailable(url: str) -> bytes:
        if "asos.py" in url:
            return ASOS
        msg = "IFS unavailable"
        raise TimeoutError(msg)

    with pytest.raises(TimeoutError, match="IFS unavailable"):
        fetch_inputs(
            [TOWN],
            start=datetime(2026, 11, 4, tzinfo=UTC),
            end=datetime(2026, 11, 6, tzinfo=UTC),
            output=tmp_path,
            request=unavailable,
        )
    assert {name: (tmp_path / name).read_bytes() for name in stored} == stored
