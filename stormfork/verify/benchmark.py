"""Compare hindcast tracks with operational forecast aids.

The comparison follows NHC verification practice: positions are verified
against the USA best track only while it is a tropical or subtropical cyclone,
and skill is measured against OCD5, NHC's no-skill reference. Late model runs
and forecaster-facing products stay in separate, labelled groups.
"""

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stormfork.cli_data import parse_time
from stormfork.errors import UserError
from stormfork.grid import great_circle_km
from stormfork.track.match import ensemble_mean
from stormfork.verify.adeck import ADeck, fetch_adeck, read_adeck

LEADS = (24, 48, 72, 96, 120)
NHC_BASINS = frozenset({"AL", "EP", "CP"})
VERIFICATION_STATUSES = frozenset({"TD", "TS", "HU", "SS", "SD"})
OCD5 = "OCD5"
OUR_MEAN = "Stormfork ensemble mean"
OUR_MEMBER = "Stormfork member 0"
LATE_LABEL = "Late, un-interpolated model runs from the same synoptic time"
CONTEXT_LABEL = "Forecaster-facing products, shown for context only"
MISSING_VALUE = -9999.0
GOOGLE_YEAR = 2025
CASE_FILE = "case.json"
BENCHMARK_FILE = "benchmark.json"

AID_DEFINITIONS = {
    "AEMN": "Global Ensemble Forecast System",
    "EEMN": "ECMWF Ensemble Prediction System (EPS)",
    "GDMN": "Google DeepMind Ensemble Mean",
    "AVNO": "Global Forecast System (FV3-GFS)",
    "EMX": "European Centre for Medium-Range Weather Forecasts",
    "OFCL": "Official NHC track forecast",
    "AVNI": "Interpolated Global Forecast System",
    "EMXI": "Interpolated European Centre for Medium-Range Weather Forecasts",
    "GDMI": "Interpolated Google DeepMind Ensemble Mean",
    "OCD5": "Combination of CLP5 and Decay-SHIFOR run on operational inputs",
    OUR_MEAN: "Eight-member mean with at least five members at each lead",
    OUR_MEMBER: "Member 0, the deterministic-like single run",
}

Role = Literal[
    "ensemble_mean", "member_0", "deterministic", "official", "interpolated", "reference"
]
SetId = Literal["ensemble_means", "single_runs", "forecaster_facing"]
GroupId = Literal["late", "forecaster_facing"]


def _utc_naive(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


class _Model(BaseModel):
    """Reject unknown keys and nonfinite numbers in published benchmark evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BestTrackPoint(_Model):
    """One USA best-track position with its cyclone status."""

    valid_time: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    status: str


class ForecastPoint(BaseModel):
    """One aid position at a forecast lead from a model cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    aid: str
    init: datetime
    lead_hours: int
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=360)


class LeadScore(_Model):
    """Homogeneous error and NHC skill for one forecaster lead."""

    lead_hours: int = Field(gt=0)
    n: int = Field(ge=0)
    error_km: float | None = None
    skill_vs_ocd5_pct: float | None = None


class AidScore(_Model):
    """One aid's labelled scores across the benchmark leads."""

    aid: str
    definition: str
    role: Role
    leads: list[LeadScore]


class ComparisonSet(_Model):
    """Aids that share one homogeneous sample and are ranked only inside it."""

    id: SetId
    label: str
    aids: list[AidScore]


class BenchmarkGroup(_Model):
    """A labelled comparison group; groups are never combined into one ranking."""

    id: GroupId
    label: str
    sets: list[ComparisonSet]


class Benchmark(_Model):
    """Published hindcast comparison, with every number the table needs."""

    case: str
    storm: str
    init_source: str
    init_times: list[datetime]
    lead_hours: list[int]
    applicable: bool
    reason: str | None = None
    groups: list[BenchmarkGroup]
    notes: list[str]


class HindcastRunSpec(_Model):
    """One model cycle and its local track table."""

    init: datetime
    tracks: str

    @field_validator("init")
    @classmethod
    def _naive_utc(cls, value: datetime) -> datetime:
        return _utc_naive(value)


class HindcastCase(_Model):
    """Local hindcast case definition, with paths relative to the case file."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    storm: str = Field(pattern=r"^[A-Z]{2}\d{2}\d{4}$")
    init_source: str = Field(min_length=1)
    ibtracs: str
    adeck: str | None = None
    runs: list[HindcastRunSpec] = Field(min_length=1)


@dataclass(frozen=True)
class RunForecasts:
    """Our ensemble mean and member 0, keyed by forecast lead."""

    init: datetime
    mean: Mapping[int, ForecastPoint]
    member_0: Mapping[int, ForecastPoint]


def storm_basin(storm: str) -> str:
    """Return the two-letter ATCF basin for a storm identifier."""
    return storm[:2].upper()


def is_nhc_basin(storm: str) -> bool:
    """Return whether NHC issues a-deck guidance for the storm's basin."""
    return storm_basin(storm) in NHC_BASINS


def is_verification_time(point: BestTrackPoint) -> bool:
    """Apply NHC's rule: verify only while the best track is tropical or subtropical."""
    return point.status.strip().upper() in VERIFICATION_STATUSES


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number) or number == MISSING_VALUE:
        return None
    return number


def load_best_track(path: str | Path, storm: str) -> dict[datetime, BestTrackPoint]:
    """Read IBTrACS USA columns for one storm, keyed by valid time.

    The standard IBTrACS CSV second row contains units; rows without a parseable
    USA position or time are ignored.

    Raises:
        ValueError: The file lacks USA columns, has no storm rows or repeats a
            time with different positions.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fieldnames = set(reader.fieldnames or ())
        required = {"ISO_TIME", "USA_LAT", "USA_LON", "USA_STATUS"}
        missing = required - fieldnames
        if missing:
            msg = f"IBTrACS file is missing columns: {', '.join(sorted(missing))}"
            raise ValueError(msg)
        identifier = storm.strip().upper()
        use_atcf_id = "USA_ATCF_ID" in fieldnames
        points: dict[datetime, BestTrackPoint] = {}
        matched = False
        for row in reader:
            if use_atcf_id and (row.get("USA_ATCF_ID") or "").strip().upper() != identifier:
                continue
            matched = True
            lat, lon = _number(row.get("USA_LAT")), _number(row.get("USA_LON"))
            time_text = (row.get("ISO_TIME") or "").strip()
            if lat is None or lon is None or not time_text:
                continue
            try:
                valid_time = parse_time(time_text)
            except ValueError:
                continue
            point = BestTrackPoint(
                valid_time=valid_time,
                lat=lat,
                lon=lon,
                status=(row.get("USA_STATUS") or "").strip().upper(),
            )
            previous = points.get(valid_time)
            if previous is not None and (previous.lat, previous.lon) != (point.lat, point.lon):
                msg = f"IBTrACS repeats {valid_time.isoformat()} with different positions"
                raise ValueError(msg)
            points[valid_time] = point
    if use_atcf_id and not matched:
        msg = f"IBTrACS has no USA rows for {identifier}"
        raise ValueError(msg)
    if not points:
        msg = f"IBTrACS has no USA positions for {identifier}"
        raise ValueError(msg)
    return points


def _forecast_index(frame: pd.DataFrame, init: datetime, aid: str) -> dict[int, ForecastPoint]:
    positions: dict[int, ForecastPoint] = {}
    for valid, lat, lon in zip(frame["valid_time"], frame["lat"], frame["lon"], strict=True):
        delta = pd.Timestamp(valid) - pd.Timestamp(init)
        hours = delta.total_seconds() / 3600
        lead = int(hours)
        if lead != hours or lead not in LEADS:
            continue
        point = ForecastPoint(
            aid=aid,
            init=init,
            lead_hours=lead,
            lat=float(lat),
            lon=float(lon),
        )
        previous = positions.get(lead)
        if previous is not None and (previous.lat, previous.lon) != (point.lat, point.lon):
            msg = f"{aid} has conflicting positions at lead {lead} for {init.isoformat()}"
            raise ValueError(msg)
        positions[lead] = point
    return positions


def forecasts_from_tracks(tracks: pd.DataFrame, init: datetime) -> RunForecasts:
    """Derive our ensemble mean and member 0 from a matched track table.

    The ensemble mean uses the existing five-of-eight-member rule; missing
    members therefore remove a lead rather than interpolate a position.

    Raises:
        ValueError: The track table lacks required columns or has ambiguous tracks.
    """
    init = _utc_naive(init)
    frame = tracks.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True).dt.tz_localize(None)
    mean = ensemble_mean(frame)
    member = frame.loc[frame["member"] == 0]
    if mean.empty and member.empty:
        msg = f"track table for {init.isoformat()} has no member 0 or usable ensemble mean"
        raise ValueError(msg)
    return RunForecasts(
        init=init,
        mean=_forecast_index(mean, init, OUR_MEAN),
        member_0=_forecast_index(member, init, OUR_MEMBER),
    )


def _adeck_index(adeck: ADeck) -> dict[str, dict[datetime, dict[int, ForecastPoint]]]:
    index: dict[str, dict[datetime, dict[int, ForecastPoint]]] = {}
    for point in adeck.points:
        by_init = index.setdefault(point.aid, {})
        by_lead = by_init.setdefault(point.init, {})
        by_lead[point.lead_hours] = ForecastPoint(
            aid=point.aid,
            init=point.init,
            lead_hours=point.lead_hours,
            lat=point.lat,
            lon=point.lon,
        )
    return index


def _skill(error: float | None, reference: float | None, n: int) -> float | None:
    if n == 0 or error is None or reference is None or reference == 0:
        return None
    return 100.0 * (1.0 - error / reference)


def _score_set(
    set_id: SetId,
    label: str,
    aids: Sequence[tuple[str, Role, Mapping[datetime, Mapping[int, ForecastPoint]]]],
    runs: Sequence[RunForecasts],
    best_track: Mapping[datetime, BestTrackPoint],
) -> ComparisonSet:
    errors: dict[str, dict[int, list[float]]] = {
        aid: {lead: [] for lead in LEADS} for aid, _role, _index in aids
    }
    for run in runs:
        for lead in LEADS:
            truth = best_track.get(run.init + timedelta(hours=lead))
            if truth is None or not is_verification_time(truth):
                continue
            positions: dict[str, ForecastPoint] = {}
            for aid, _role, index in aids:
                point = index.get(run.init, {}).get(lead)
                if point is None:
                    break
                positions[aid] = point
            else:
                for aid, point in positions.items():
                    errors[aid][lead].append(
                        float(great_circle_km(point.lat, point.lon, truth.lat, truth.lon))
                    )
    scored: list[AidScore] = []
    for aid, role, _index in aids:
        lead_scores: list[LeadScore] = []
        for lead in LEADS:
            sample = errors[aid][lead]
            n = len(sample)
            error = fmean(sample) if n else None
            reference_sample = errors[OCD5][lead]
            reference = fmean(reference_sample) if reference_sample else None
            lead_scores.append(
                LeadScore(
                    lead_hours=lead,
                    n=n,
                    error_km=error,
                    skill_vs_ocd5_pct=_skill(error, reference, n),
                )
            )
        scored.append(
            AidScore(
                aid=aid,
                definition=AID_DEFINITIONS[aid],
                role=role,
                leads=lead_scores,
            )
        )
    return ComparisonSet(id=set_id, label=label, aids=scored)


def _notes(init_source: str, init_count: int, *, google: bool) -> list[str]:
    notes = [
        f"{init_count} inits are a case study, not a skill estimate.",
        (
            "An 8-member mean is not an equal comparison with the 31-member GEFS, "
            "51-member ECMWF or Google's operational ensemble means."
        ),
        (
            "Skill is 100 x (1 - error / OCD5 error), NHC's convention; it is null "
            "where the homogeneous OCD5 error is zero or n = 0."
        ),
        (
            "Late model runs and forecaster-facing products are separate groups; "
            "the second group was issued in real time from older guidance and is context only."
        ),
        (
            "Aid definitions are quoted from NHC's model summary; the 2025 verification "
            "report identifies GDMN and GDMI as the Google DeepMind ensemble mean."
        ),
    ]
    if google:
        notes.append(
            "GDMN and GDMI are present only for 2025 inits, so their homogeneous "
            "samples contain 2025 inits only."
        )
    if init_source.strip().lower() == "era5":
        notes.append(
            "The ERA5 00 UTC analysis assimilates observations up to 09 UTC "
            "(its 12-hour window), an advantage no operational aid had."
        )
    return notes


def build_benchmark(  # noqa: PLR0913
    *,
    case: str,
    storm: str,
    init_source: str,
    runs: Sequence[RunForecasts],
    adeck: ADeck | None,
    best_track: Mapping[datetime, BestTrackPoint],
) -> Benchmark:
    """Build the two-group benchmark from ready forecasts and best-track truth.

    When no a-deck is available, the storm is outside NHC's basins or has no
    public guidance, so the benchmark reports that it does not apply instead of
    inventing comparisons.

    Raises:
        ValueError: Runs repeat a model cycle or lack our member-0 track.
    """
    inits = [run.init for run in runs]
    if len(set(inits)) != len(inits):
        msg = "each hindcast init must appear once"
        raise ValueError(msg)
    if adeck is None:
        return Benchmark(
            case=case,
            storm=storm,
            init_source=init_source,
            init_times=sorted(inits),
            lead_hours=list(LEADS),
            applicable=False,
            reason="No a-deck is available, so the benchmark does not apply to this storm.",
            groups=[],
            notes=[],
        )
    google = any(init.year == GOOGLE_YEAR for init in inits)
    deck = _adeck_index(adeck)
    ours_mean: dict[datetime, Mapping[int, ForecastPoint]] = {run.init: run.mean for run in runs}
    ours_member: dict[datetime, Mapping[int, ForecastPoint]] = {
        run.init: run.member_0 for run in runs
    }
    ensemble_aids: list[tuple[str, Role, Mapping[datetime, Mapping[int, ForecastPoint]]]] = [
        (OUR_MEAN, "ensemble_mean", ours_mean),
        ("AEMN", "ensemble_mean", deck.get("AEMN", {})),
        ("EEMN", "ensemble_mean", deck.get("EEMN", {})),
    ]
    if google:
        ensemble_aids.append(("GDMN", "ensemble_mean", deck.get("GDMN", {})))
    ensemble_aids.append((OCD5, "reference", deck.get(OCD5, {})))
    single_aids: list[tuple[str, Role, Mapping[datetime, Mapping[int, ForecastPoint]]]] = [
        (OUR_MEMBER, "member_0", ours_member),
        ("AVNO", "deterministic", deck.get("AVNO", {})),
        ("EMX", "deterministic", deck.get("EMX", {})),
        (OCD5, "reference", deck.get(OCD5, {})),
    ]
    context_aids: list[tuple[str, Role, Mapping[datetime, Mapping[int, ForecastPoint]]]] = [
        ("OFCL", "official", deck.get("OFCL", {})),
        ("AVNI", "interpolated", deck.get("AVNI", {})),
        ("EMXI", "interpolated", deck.get("EMXI", {})),
    ]
    if google:
        context_aids.append(("GDMI", "interpolated", deck.get("GDMI", {})))
    context_aids.append((OCD5, "reference", deck.get(OCD5, {})))
    groups = [
        BenchmarkGroup(
            id="late",
            label=LATE_LABEL,
            sets=[
                _score_set("ensemble_means", "Ensemble means", ensemble_aids, runs, best_track),
                _score_set("single_runs", "Single runs", single_aids, runs, best_track),
            ],
        ),
        BenchmarkGroup(
            id="forecaster_facing",
            label=CONTEXT_LABEL,
            sets=[
                _score_set(
                    "forecaster_facing",
                    "Official and interpolated aids",
                    context_aids,
                    runs,
                    best_track,
                )
            ],
        ),
    ]
    return Benchmark(
        case=case,
        storm=storm,
        init_source=init_source,
        init_times=sorted(inits),
        lead_hours=list(LEADS),
        applicable=True,
        groups=groups,
        notes=_notes(init_source, len(runs), google=google),
    )


def load_case(path: str | Path) -> tuple[HindcastCase, Path]:
    """Load a case directory or its case.json file.

    Raises:
        UserError: The case path does not exist.
        ValueError: The case file is invalid.
    """
    path = Path(path)
    candidate = path if path.name == CASE_FILE else path / CASE_FILE
    if path.is_file():
        candidate = path
    if not candidate.exists():
        msg = f'hindcast case "{path}" has no {CASE_FILE}'
        raise UserError(msg, "supply a case directory containing case.json", 2)
    return HindcastCase.model_validate_json(candidate.read_bytes()), candidate.parent


def _local(directory: Path, relative: str) -> Path:
    path = directory / relative
    if not path.resolve().is_relative_to(directory.resolve()):
        msg = f"case path escapes the case directory: {relative}"
        raise ValueError(msg)
    return path


def _adeck_for(case: HindcastCase, directory: Path) -> ADeck | None:
    if not is_nhc_basin(case.storm):
        return None
    if case.adeck is not None:
        return read_adeck(_local(directory, case.adeck))
    current_year = datetime.now(UTC).year
    if any(run.init.year == current_year for run in case.runs):
        return fetch_adeck(case.storm)
    msg = f"no a-deck is configured for {case.id}"
    raise UserError(msg, "set adeck in case.json or benchmark a current-year storm", 2)


def benchmark_case(case_path: str | Path, output: Path | None = None) -> Path:
    """Read a hindcast case, score it and write benchmark.json.

    Raises:
        UserError: The case or a required input is absent.
        ValueError: An input violates the case or track contract.
    """
    case, directory = load_case(Path(case_path))
    if is_nhc_basin(case.storm):
        adeck = _adeck_for(case, directory)
        best_track = load_best_track(_local(directory, case.ibtracs), case.storm)
        runs = [
            forecasts_from_tracks(pd.read_csv(_local(directory, run.tracks)), run.init)
            for run in case.runs
        ]
    else:
        adeck = None
        best_track = {}
        runs = [RunForecasts(spec.init, {}, {}) for spec in case.runs]
    benchmark = build_benchmark(
        case=case.id,
        storm=case.storm,
        init_source=case.init_source,
        runs=runs,
        adeck=adeck,
        best_track=best_track,
    )
    destination = output or directory / BENCHMARK_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def resolve_case(name: str) -> Path:
    """Resolve a case id under the local hindcast directories.

    Raises:
        UserError: No candidate directory exists.
    """
    candidate = Path(name)
    candidates = (candidate, Path("hindcasts") / name, Path("data") / "hindcasts" / name)
    for path in candidates:
        if path.exists():
            return path
    msg = f'hindcast case "{name}" does not exist'
    raise UserError(msg, "supply a case directory or add it under hindcasts", 2)
