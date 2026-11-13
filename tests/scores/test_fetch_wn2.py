from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from stormfork.scores.fetch import (
    METADATA_URL,
    EnsemblePayload,
    RunMetadata,
    UnitMismatchError,
    collect_wn2,
    parse_wn2,
    wn2_url,
)
from stormfork.scores.towns import load_towns

FIXTURES = Path("tests/fixtures")
TOWNS = load_towns(Path("data/towns.json"))
INIT = RunMetadata.model_validate_json((FIXTURES / "wn2_meta.json").read_bytes()).init_time


def test_request_has_units_and_full_run_window():
    query = parse_qs(urlsplit(wn2_url(TOWNS[1], INIT)).query)
    assert query["wind_speed_unit"] == ["ms"]
    assert query["temperature_unit"] == ["celsius"]
    assert query["start_hour"] == ["2026-11-05T00:00"]
    assert query["end_hour"] == ["2026-11-12T00:00"]
    assert query["models"] == ["google_weathernext2_ensemble"]
    assert "temporal_resolution" not in query


def test_recorded_run_preserves_all_leads_and_member_zero():
    body = (FIXTURES / "wn2_ensemble.json").read_bytes()
    rows = parse_wn2(body, TOWNS[1], INIT)
    assert len(rows) == 29 * 3
    wind = [row for row in rows if row.variable == "wind_speed_10m"]
    assert wind[0].member_0 == 1.3
    assert wind[0].town == "KMIA"
    payload = EnsemblePayload.model_validate_json(body)
    first_members = [
        payload.hourly["wind_speed_10m" if member == 0 else f"wind_speed_10m_member{member:02d}"][0]
        for member in range(64)
    ]
    numeric_members = [value for value in first_members if isinstance(value, float)]
    assert len(numeric_members) == 64
    assert wind[0].ens_mean == pytest.approx(fmean(numeric_members))
    assert wind[0].ens_mean != wind[0].member_0
    assert [row.valid_time for row in wind] == [
        INIT + timedelta(hours=lead) for lead in range(0, 169, 6)
    ]
    assert all(row.init_time == INIT for row in rows)


def test_recorded_inconsistent_six_hourly_response_is_rejected():
    body = (FIXTURES / "wn2_inconsistent_six_hourly.json").read_bytes()
    with pytest.raises(ValueError, match="hourly intervals"):
        parse_wn2(body, TOWNS[1], INIT)


@pytest.mark.parametrize(
    ("key", "unit"),
    [
        ("wind_speed_10m", "km/h"),
        ("wind_speed_10m_member63", "km/h"),
        ("temperature_2m", "K"),
        ("pressure_msl", "Pa"),
    ],
)
def test_wrong_units_raise(key: str, unit: str):
    payload = EnsemblePayload.model_validate_json((FIXTURES / "wn2_ensemble.json").read_bytes())
    payload.hourly_units[key] = unit
    with pytest.raises(UnitMismatchError, match=key):
        parse_wn2(payload.model_dump_json().encode(), TOWNS[1], INIT)


@pytest.mark.parametrize(
    "damage", ["null", "missing_member", "short_member", "missing_time", "offset"]
)
def test_incomplete_or_local_time_response_is_rejected(damage: str):
    payload = EnsemblePayload.model_validate_json((FIXTURES / "wn2_ensemble.json").read_bytes())
    if damage == "null":
        payload.hourly["wind_speed_10m_member01"][1] = None
    elif damage == "missing_member":
        del payload.hourly["wind_speed_10m_member63"]
    elif damage == "short_member":
        payload.hourly["temperature_2m_member01"].pop()
    elif damage == "missing_time":
        payload.hourly["time"].pop()
    else:
        payload.utc_offset_seconds = 3600
    with pytest.raises(ValueError, match=r"missing|incomplete|168 h|UTC"):
        parse_wn2(payload.model_dump_json().encode(), TOWNS[1], INIT)


def test_collector_writes_complete_parquet_for_all_towns(tmp_path: Path):
    requests: list[str] = []

    def recorded_request(url: str) -> bytes:
        requests.append(url)
        fixture = "wn2_meta.json" if url == METADATA_URL else "wn2_ensemble.json"
        return (FIXTURES / fixture).read_bytes()

    destination = collect_wn2(
        TOWNS, tmp_path, now=INIT + timedelta(hours=18), request=recorded_request
    )
    frame = pd.read_parquet(destination)
    assert destination == tmp_path / "wn2/2026-11-05-00.parquet"
    assert list(frame.columns) == [
        "town",
        "init_time",
        "valid_time",
        "variable",
        "ens_mean",
        "member_0",
    ]
    assert len(frame) == 24 * 29 * 3
    assert set(frame["town"]) == {town.icao for town in TOWNS}
    assert str(frame["init_time"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["valid_time"].dtype) == "datetime64[ns, UTC]"
    assert len(requests) == 26
    assert requests[0] == requests[-1] == METADATA_URL
    assert len(set(requests[1:-1])) == 24
    assert not destination.with_suffix(".parquet.tmp").exists()


def test_run_change_aborts_without_output(tmp_path: Path):
    calls = 0

    def changing_request(url: str) -> bytes:
        nonlocal calls
        if url != METADATA_URL:
            return (FIXTURES / "wn2_ensemble.json").read_bytes()
        calls += 1
        metadata = RunMetadata.model_validate_json((FIXTURES / "wn2_meta.json").read_bytes())
        if calls == 2:
            metadata.last_run_initialisation_time += 12 * 3600
        return metadata.model_dump_json().encode()

    with pytest.raises(ValueError, match="changed during collection"):
        collect_wn2(TOWNS[:1], tmp_path, now=INIT + timedelta(hours=18), request=changing_request)
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("now", [INIT - timedelta(hours=1), INIT + timedelta(days=2)])
def test_unavailable_run_is_rejected(tmp_path: Path, now: datetime):
    def metadata_only(_url: str) -> bytes:
        return (FIXTURES / "wn2_meta.json").read_bytes()

    with pytest.raises(ValueError, match=r"replicating|stale"):
        collect_wn2(TOWNS, tmp_path, now=now, request=metadata_only)
    assert not list(tmp_path.iterdir())


def test_recently_available_run_waits_for_replication(tmp_path: Path):
    metadata = RunMetadata.model_validate_json((FIXTURES / "wn2_meta.json").read_bytes())

    def metadata_only(_url: str) -> bytes:
        return metadata.model_dump_json().encode()

    now = datetime.fromtimestamp(metadata.last_run_availability_time, UTC) + timedelta(minutes=9)
    with pytest.raises(ValueError, match="replicating"):
        collect_wn2(TOWNS, tmp_path, now=now, request=metadata_only)


def test_stale_metadata_names_a_concrete_retry_without_requesting_forecasts(tmp_path: Path):
    requested: list[str] = []

    def metadata_only(url: str) -> bytes:
        requested.append(url)
        return (FIXTURES / "wn2_meta.json").read_bytes()

    with pytest.raises(ValueError, match="stale") as raised:
        collect_wn2(TOWNS, tmp_path, now=INIT + timedelta(days=2), request=metadata_only)
    assert getattr(raised.value, "next_step", "") == (
        "retry this command in 30 minutes; see docs/scores.md for metadata checks"
    )
    assert requested == [METADATA_URL]
    assert not list(tmp_path.rglob("*.parquet"))
