import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stormfork.grid import EARTH_RADIUS_KM
from stormfork.verify.adeck import ADeck, ADeckPoint
from stormfork.verify.benchmark import (
    AID_DEFINITIONS,
    LEADS,
    OUR_MEAN,
    OUR_MEMBER,
    Benchmark,
    BestTrackPoint,
    ForecastPoint,
    RunForecasts,
    build_benchmark,
    forecasts_from_tracks,
    is_nhc_basin,
    is_verification_time,
    load_best_track,
)
from tests.test_cli import invoke

STORM = "AL142024"
FIRST_INIT = datetime.fromisoformat("2024-10-05T00:00:00")
SECOND_INIT = datetime.fromisoformat("2024-10-06T00:00:00")
TRUTH_LAT = 10.0
TRUTH_LON = -60.0
OFFSETS = {
    "AEMN": 1.0,
    "EEMN": 1.5,
    "GDMN": 0.8,
    "AVNO": 0.7,
    "EMX": 1.1,
    "OFCL": 0.4,
    "AVNI": 0.6,
    "EMXI": 0.9,
    "GDMI": 0.3,
    "OCD5": 2.0,
}
DEGREE_KM = EARTH_RADIUS_KM * math.pi / 180


def make_run(
    init: datetime, *, mean_offset: float = 0.5, member_offset: float = 0.5
) -> RunForecasts:
    mean = {
        lead: ForecastPoint(
            aid=OUR_MEAN,
            init=init,
            lead_hours=lead,
            lat=TRUTH_LAT + mean_offset,
            lon=TRUTH_LON,
        )
        for lead in LEADS
    }
    member = {
        lead: ForecastPoint(
            aid=OUR_MEMBER,
            init=init,
            lead_hours=lead,
            lat=TRUTH_LAT + member_offset,
            lon=TRUTH_LON,
        )
        for lead in LEADS
    }
    return RunForecasts(init=init, mean=mean, member_0=member)


def make_adeck(
    inits: tuple[datetime, ...],
    *,
    missing: frozenset[tuple[str, datetime, int]] = frozenset(),
    offsets: dict[str, float] = OFFSETS,
) -> ADeck:
    points = [
        ADeckPoint(
            storm=STORM,
            aid=aid,
            init=init,
            lead_hours=lead,
            lat=TRUTH_LAT + offset,
            lon=TRUTH_LON,
        )
        for init in inits
        for lead in LEADS
        for aid, offset in offsets.items()
        if (aid, init, lead) not in missing and not (aid in {"GDMN", "GDMI"} and init.year != 2025)
    ]
    return ADeck(storm=STORM, points=tuple(points))


def truth(init: datetime, lead: int, status: str = "TS") -> BestTrackPoint:
    return BestTrackPoint(
        valid_time=init + timedelta(hours=lead),
        lat=TRUTH_LAT,
        lon=TRUTH_LON,
        status=status,
    )


def make_truth(
    inits: tuple[datetime, ...], statuses: dict[tuple[datetime, int], str] | None = None
):
    statuses = statuses or {}
    return {
        init + timedelta(hours=lead): truth(init, lead, statuses.get((init, lead), "TS"))
        for init in inits
        for lead in LEADS
    }


def benchmark(
    inits: tuple[datetime, ...] = (FIRST_INIT, SECOND_INIT),
    *,
    missing: frozenset[tuple[str, datetime, int]] = frozenset(),
    truth_points: dict[datetime, BestTrackPoint] | None = None,
) -> Benchmark:
    runs = [make_run(init) for init in inits]
    return build_benchmark(
        case="hindcast_milton_2024",
        storm=STORM,
        init_source="era5",
        runs=runs,
        adeck=make_adeck(inits, missing=missing),
        best_track=truth_points if truth_points is not None else make_truth(inits),
    )


def ensemble_scores(result: Benchmark) -> dict[str, list[tuple[int, float | None, int]]]:
    aids = result.groups[0].sets[0].aids
    return {
        aid.aid: [(lead.lead_hours, lead.error_km, lead.n) for lead in aid.leads] for aid in aids
    }


def test_homogeneous_error_and_nhc_skill_are_hand_computable():
    result = benchmark()
    assert result.applicable
    assert result.lead_hours == list(LEADS)
    scores = ensemble_scores(result)
    assert scores[OUR_MEAN][0] == (24, pytest.approx(0.5 * DEGREE_KM), 2)
    assert scores["AEMN"][0] == (24, pytest.approx(1.0 * DEGREE_KM), 2)
    assert scores["EEMN"][0] == (24, pytest.approx(1.5 * DEGREE_KM), 2)
    assert scores["OCD5"][0] == (24, pytest.approx(2.0 * DEGREE_KM), 2)
    ours = result.groups[0].sets[0].aids[0]
    assert ours.leads[0].skill_vs_ocd5_pct == pytest.approx(75.0)
    assert result.groups[0].sets[0].aids[-1].leads[0].skill_vs_ocd5_pct == pytest.approx(0.0)


def test_homogeneous_sample_drops_an_init_when_any_aid_is_missing():
    result = benchmark(missing=frozenset({("EEMN", SECOND_INIT, 48)}))
    scores = ensemble_scores(result)
    assert {lead: n for lead, _error, n in scores[OUR_MEAN]} == {
        24: 2,
        48: 1,
        72: 2,
        96: 2,
        120: 2,
    }
    assert {lead: n for lead, _error, n in scores["AEMN"]}[48] == 1


def test_best_track_status_controls_verification():
    points = make_truth((FIRST_INIT, SECOND_INIT), {(SECOND_INIT, 24): "EX"})
    result = benchmark(truth_points=points)
    scores = ensemble_scores(result)
    assert {lead: n for lead, _error, n in scores[OUR_MEAN]}[24] == 1
    assert is_verification_time(truth(FIRST_INIT, 24, "HU"))
    assert is_verification_time(truth(FIRST_INIT, 24, "SS"))
    assert not is_verification_time(truth(FIRST_INIT, 24, "EX"))
    assert not is_verification_time(truth(FIRST_INIT, 24, "LO"))


def test_zero_ocd5_error_and_empty_samples_publish_null_skill():
    zero_reference = make_adeck((FIRST_INIT, SECOND_INIT), offsets={**OFFSETS, "OCD5": 0.0})
    runs = [make_run(FIRST_INIT), make_run(SECOND_INIT)]
    result = build_benchmark(
        case="zero",
        storm=STORM,
        init_source="hres_sample",
        runs=runs,
        adeck=zero_reference,
        best_track=make_truth((FIRST_INIT, SECOND_INIT)),
    )
    for aid in result.groups[0].sets[0].aids:
        assert all(lead.skill_vs_ocd5_pct is None for lead in aid.leads)
    empty = benchmark(truth_points={})
    for aid in empty.groups[0].sets[0].aids:
        assert all(lead.n == 0 for lead in aid.leads)
        assert all(lead.error_km is None and lead.skill_vs_ocd5_pct is None for lead in aid.leads)


def test_groups_and_sets_are_never_mixed():
    result = benchmark()
    assert [group.id for group in result.groups] == ["late", "forecaster_facing"]
    late = result.groups[0]
    assert [comparison.id for comparison in late.sets] == ["ensemble_means", "single_runs"]
    assert [comparison.id for comparison in result.groups[1].sets] == ["forecaster_facing"]
    ensemble_ids = {aid.aid for aid in late.sets[0].aids}
    single_ids = {aid.aid for aid in late.sets[1].aids}
    context_ids = {aid.aid for aid in result.groups[1].sets[0].aids}
    assert "OFCL" not in ensemble_ids
    assert "OFCL" not in single_ids
    assert "AEMN" not in single_ids
    assert "AEMN" not in context_ids
    assert OUR_MEAN not in context_ids
    assert OUR_MEMBER not in context_ids
    assert "OCD5" in ensemble_ids
    assert "OCD5" in single_ids
    assert "OCD5" in context_ids


def test_google_aids_only_join_2025_samples():
    campaign_2024 = benchmark()
    assert all(
        aid.aid not in {"GDMN", "GDMI"}
        for group in campaign_2024.groups
        for s in group.sets
        for aid in s.aids
    )
    inits = (
        datetime.fromisoformat("2025-10-01T00:00:00"),
        datetime.fromisoformat("2025-10-02T00:00:00"),
    )
    campaign_2025 = benchmark(inits)
    ensemble_ids = {aid.aid for aid in campaign_2025.groups[0].sets[0].aids}
    context_ids = {aid.aid for aid in campaign_2025.groups[1].sets[0].aids}
    assert "GDMN" in ensemble_ids
    assert "GDMI" in context_ids
    assert {lead.n for lead in campaign_2025.groups[0].sets[0].aids[0].leads} == {2}


def test_mixed_years_restrict_google_aids_to_their_homogeneous_sample():
    inits = (
        datetime.fromisoformat("2024-10-05T00:00:00"),
        datetime.fromisoformat("2025-10-01T00:00:00"),
    )
    result = benchmark(inits)
    scores = ensemble_scores(result)
    assert {lead: n for lead, _error, n in scores[OUR_MEAN]} == dict.fromkeys(LEADS, 1)


def test_aid_definitions_are_quoted_from_the_operational_lists():
    assert AID_DEFINITIONS["AEMN"] == "Global Ensemble Forecast System"
    assert AID_DEFINITIONS["EEMN"] == "ECMWF Ensemble Prediction System (EPS)"
    assert AID_DEFINITIONS["AVNO"] == "Global Forecast System (FV3-GFS)"
    assert AID_DEFINITIONS["GDMN"] == "Google DeepMind Ensemble Mean"
    assert AID_DEFINITIONS["OCD5"] == (
        "Combination of CLP5 and Decay-SHIFOR run on operational inputs"
    )


def test_page_notes_state_the_case_study_and_era5_advantage():
    notes = " ".join(benchmark().notes)
    assert "2 inits are a case study, not a skill estimate." in notes
    assert "ERA5 00 UTC analysis assimilates observations up to 09 UTC" in notes
    assert "31-member GEFS" in notes


def test_no_adeck_reports_not_applicable():
    result = build_benchmark(
        case="hindcast_west_pacific",
        storm="WP012025",
        init_source="era5",
        runs=[],
        adeck=None,
        best_track={},
    )
    assert not result.applicable
    assert result.groups == []
    assert result.reason is not None
    assert "does not apply" in result.reason
    assert not is_nhc_basin("WP012025")
    assert is_nhc_basin("EP112024")


def test_forecasts_from_tracks_applies_the_five_of_eight_rule():
    frame = synthetic_tracks(8)
    run = forecasts_from_tracks(frame, FIRST_INIT)
    assert sorted(run.mean) == list(LEADS)
    assert run.mean[24].lat == pytest.approx(10.5)
    sparse = forecasts_from_tracks(synthetic_tracks(4), FIRST_INIT)
    assert sparse.mean == {}
    assert sorted(sparse.member_0) == list(LEADS)


def test_best_track_reads_usa_columns_and_skips_missing_positions(tmp_path: Path):
    path = tmp_path / "ibtracs.csv"
    path.write_text(
        "SID,ISO_TIME,USA_ATCF_ID,USA_LAT,USA_LON,USA_STATUS\n"
        "AL142024,2024-10-05 00:00:00,AL142024,10.0,-60.0,TS\n"
        "AL142024,2024-10-05 06:00:00,AL142024,-9999,-9999,TS\n"
        "AL142024,2024-10-05 12:00:00,AL142024,10.5,-60.1,EX\n"
        "EP112024,2024-10-05 00:00:00,EP112024,12.0,-110.0,TS\n"
    )
    points = load_best_track(path, STORM)
    assert sorted(points) == [
        datetime.fromisoformat("2024-10-05T00:00:00"),
        datetime.fromisoformat("2024-10-05T12:00:00"),
    ]
    assert points[datetime.fromisoformat("2024-10-05T12:00:00")].status == "EX"
    with pytest.raises(ValueError, match="no USA rows"):
        load_best_track(path, "AL132024")


def synthetic_tracks(members: int) -> pd.DataFrame:
    rows = [
        {
            "member": member,
            "step": step,
            "valid_time": FIRST_INIT + timedelta(hours=step * 6),
            "lat": 10.5,
            "lon": TRUTH_LON,
            "mslp_pa": 98000.0,
            "wind_ms": 40.0,
            "track_id": "target",
        }
        for member in range(members)
        for step in range(21)
    ]
    return pd.DataFrame(rows)


def atcf_coordinate(value: float, *, latitude: bool) -> str:
    magnitude = round(abs(value) * 10)
    if latitude:
        return f"{magnitude:03d}{'N' if value >= 0 else 'S'}"
    return f"{magnitude:03d}{'E' if value >= 0 else 'W'}"


def truth_lat(valid_time: datetime) -> float:
    return TRUTH_LAT + 0.1 * (valid_time - FIRST_INIT).total_seconds() / (6 * 3600)


def write_case(directory: Path, *, storm: str = STORM) -> None:
    inits = (FIRST_INIT, SECOND_INIT)
    for init in inits:
        tracks = [
            {
                "member": member,
                "step": step,
                "valid_time": init + timedelta(hours=step * 6),
                "lat": truth_lat(init + timedelta(hours=step * 6)) + 0.5,
                "lon": TRUTH_LON,
                "mslp_pa": 98000.0,
                "wind_ms": 40.0,
                "track_id": "target",
            }
            for member in range(8)
            for step in range(21)
        ]
        pd.DataFrame(tracks).to_csv(directory / f"tracks-{init:%Y%m%d%H}.csv", index=False)
    adeck_lines = [
        f"AL, 14, {init:%Y%m%d%H}, 03, {aid}, {lead:4d}, "
        f"{atcf_coordinate(truth_lat(init + timedelta(hours=lead)) + offset, latitude=True)}, "
        f"{atcf_coordinate(TRUTH_LON, latitude=False)}, 25, 1009, XX"
        for init in inits
        for lead in LEADS
        for aid, offset in OFFSETS.items()
    ]
    (directory / "adeck.dat").write_text("\n".join(adeck_lines) + "\n")
    best = [
        {
            "SID": STORM,
            "ISO_TIME": (init + timedelta(hours=hours)).isoformat(sep=" "),
            "USA_ATCF_ID": storm,
            "USA_LAT": truth_lat(init + timedelta(hours=hours)),
            "USA_LON": TRUTH_LON,
            "USA_STATUS": "TS",
        }
        for init in inits
        for hours in range(0, 121, 6)
    ]
    pd.DataFrame(best).to_csv(directory / "ibtracs.csv", index=False)
    (directory / "case.json").write_text(
        json.dumps(
            {
                "id": "hindcast_milton_2024",
                "storm": storm,
                "init_source": "era5",
                "ibtracs": "ibtracs.csv",
                "adeck": "adeck.dat",
                "runs": [
                    {
                        "init": init.isoformat(),
                        "tracks": f"tracks-{init:%Y%m%d%H}.csv",
                    }
                    for init in inits
                ],
            }
        )
    )


def test_cli_benchmark_writes_benchmark_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    write_case(tmp_path)
    invoke(monkeypatch, "benchmark", str(tmp_path))
    result = Benchmark.model_validate_json((tmp_path / "benchmark.json").read_bytes())
    assert result.case == "hindcast_milton_2024"
    assert result.init_times == [FIRST_INIT, SECOND_INIT]
    ours = result.groups[0].sets[0].aids[0]
    assert ours.leads[0].n == 2
    assert ours.leads[0].error_km == pytest.approx(0.5 * DEGREE_KM)
    assert ours.leads[0].skill_vs_ocd5_pct == pytest.approx(75.0)


def test_cli_outside_nhc_basins_reports_benchmark_does_not_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "id": "hindcast_west_pacific",
                "storm": "WP012025",
                "init_source": "era5",
                "ibtracs": "missing.csv",
                "runs": [{"init": "2025-10-01T00:00:00", "tracks": "missing.csv"}],
            }
        )
    )
    invoke(monkeypatch, "benchmark", str(tmp_path))
    result = Benchmark.model_validate_json((tmp_path / "benchmark.json").read_bytes())
    assert not result.applicable
    assert result.reason is not None
    assert "does not apply" in result.reason
