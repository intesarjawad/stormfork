import math
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from http import HTTPStatus
from pathlib import Path
from statistics import fmean
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stormfork.downloads import open_https
from stormfork.errors import UserError
from stormfork.scores.metrics import Observation
from stormfork.scores.towns import Town

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
METADATA_URL = (
    "https://ensemble-api.open-meteo.com/data/google_weathernext2_ensemble/static/meta.json"
)
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLE_UNITS = (("temperature_2m", "°C"), ("pressure_msl", "hPa"), ("wind_speed_10m", "m/s"))
PREVIOUS_RUN_MODELS = (
    "ecmwf_ifs025",
    "gfs025",
    "icon_seamless",
    "ecmwf_aifs025_single",
)
PREVIOUS_RUN_VARIABLES = (
    ("temperature_2m", "°C"),
    ("relative_humidity_2m", "%"),
    ("dew_point_2m", "°C"),
    ("precipitation", "mm"),
    ("pressure_msl", "hPa"),
    ("cloud_cover", "%"),
    ("wind_speed_10m", "m/s"),
    ("wind_gusts_10m", "m/s"),
    ("wind_direction_10m", "°"),
)
PREVIOUS_RUN_LEAD_DAYS = 1
SCORING_HOURS = (0, 6, 12, 18)
CHUNK_DAYS = 90
WN2_PROBE_DAYS = 7
# Open-Meteo's free tier allows 10,000 weighted calls a day; the archive keeps a margin.
DAILY_CALL_LIMIT = 8000.0
CALL_VARIABLE_STEP = 10
CALL_DAY_STEP = 14
OBSERVATION_WINDOW = timedelta(minutes=30)
ENSEMBLE_SIZE = 64
HORIZON_HOURS = 168
STEP_HOURS = 6
HTTP_TIMEOUT_SECONDS = 30
RETRY_DELAYS_SECONDS = (1, 2)
FORECAST_COLUMNS = ("town", "init_time", "valid_time", "variable", "ens_mean", "member_0")
PREVIOUS_RUN_COLUMNS = ("town", "model", "variable", "valid_time", "lead_days", "value")
OBSERVATION_COLUMNS = ("town", "variable", "valid_time", "value")
# Open-Meteo documents ten minutes for its replicated servers to converge.
REPLICATION_DELAY = timedelta(minutes=10)


class UnitMismatchError(ValueError):
    """The upstream response uses units incompatible with the scoreboard."""


class RunNotReadyError(ValueError):
    """Unavailable upstream data, with a timed retry for callers."""

    def __init__(self, message: str, next_step: str) -> None:
        super().__init__(message)
        self.next_step = next_step


class RunMetadata(BaseModel):
    """Open-Meteo's advertised model initialization and availability timestamps."""

    last_run_initialisation_time: int = Field(strict=True)
    last_run_availability_time: int = Field(strict=True)

    @property
    def init_time(self) -> datetime:
        """Return the UTC initialization, restricted to the published WN2 cycles."""
        init = datetime.fromtimestamp(self.last_run_initialisation_time, UTC)
        if init.hour not in (0, 12) or init.minute or init.second:
            msg = f"unexpected WN2 initialization: {init.isoformat()}"
            raise ValueError(msg)
        return init


class EnsemblePayload(BaseModel):
    """The JSON fields needed to validate a complete ensemble response."""

    model_config = ConfigDict(strict=True)
    utc_offset_seconds: int = Field(strict=True)
    hourly_units: dict[str, str]
    hourly: dict[str, list[float | str | None]]


class PreviousRunsPayload(BaseModel):
    """The JSON fields needed to validate a Previous Runs response."""

    model_config = ConfigDict(strict=True)
    utc_offset_seconds: int = Field(strict=True)
    hourly_units: dict[str, str]
    hourly: dict[str, list[float | str | None]]


class ForecastRow(BaseModel):
    """One variable and valid time, retaining both forecast comparisons."""

    town: str
    init_time: datetime
    valid_time: datetime
    variable: str
    ens_mean: float
    member_0: float


class PreviousRunRow(BaseModel):
    """One archived forecast at a fixed lead, ready for scoring against observations."""

    town: str
    model: str
    variable: str
    valid_time: datetime
    lead_days: int
    value: float


class BackfillDay(BaseModel):
    """The Open-Meteo calls one backfill run needs for one stored chunk."""

    model_config = ConfigDict(frozen=True)
    start: date
    end: date
    calls: float


def wn2_url(town: Town, init_time: datetime) -> str:
    """Request hourly data across the advertised initialization's full horizon."""
    query = urlencode(
        {
            "latitude": town.lat,
            "longitude": town.lon,
            "models": "google_weathernext2_ensemble",
            "hourly": ",".join(variable for variable, _unit in VARIABLE_UNITS),
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "timezone": "GMT",
            "start_hour": init_time.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": (init_time + timedelta(hours=HORIZON_HOURS)).strftime("%Y-%m-%dT%H:%M"),
        }
    )
    return f"{ENSEMBLE_URL}?{query}"


def _finite_member(values: Sequence[float | str | None], key: str) -> list[float]:
    result: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            msg = f"{key} has missing or nonfinite data; retry when the full run is available"
            raise ValueError(msg)
        result.append(float(value))
    return result


def parse_wn2(body: bytes, town: Town, init_time: datetime) -> list[ForecastRow]:
    """Validate the hourly response and retain six-hourly ensemble comparisons."""
    payload = EnsemblePayload.model_validate_json(body)
    if payload.utc_offset_seconds != 0:
        msg = "WN2 response must use UTC"
        raise ValueError(msg)
    times = [
        (init_time + timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M")
        for lead in range(HORIZON_HOURS + 1)
    ]
    if payload.hourly.get("time") != times:
        msg = "WN2 response must cover initialization through 168 h at hourly intervals"
        raise ValueError(msg)
    rows: list[ForecastRow] = []
    for variable, expected_unit in VARIABLE_UNITS:
        members: list[list[float]] = []
        for member in range(ENSEMBLE_SIZE):
            key = variable if member == 0 else f"{variable}_member{member:02d}"
            if payload.hourly_units.get(key) != expected_unit:
                msg = f"{key}: expected {expected_unit}, got {payload.hourly_units.get(key)!r}"
                raise UnitMismatchError(msg)
            values = _finite_member(payload.hourly.get(key, []), key)
            if len(values) != len(times):
                msg = f"{key}: incomplete WN2 member horizon"
                raise ValueError(msg)
            members.append(values)
        rows.extend(
            ForecastRow(
                town=town.icao,
                init_time=init_time,
                valid_time=init_time + timedelta(hours=index),
                variable=variable,
                ens_mean=fmean(member[index] for member in members),
                member_0=members[0][index],
            )
            for index in range(0, len(times), STEP_HOURS)
        )
    return rows


def weighted_calls(variables: int, days: int) -> float:
    """Count Open-Meteo's fractional calls for one request, as its pricing page defines them."""
    return max(1.0, variables / CALL_VARIABLE_STEP) * max(1.0, days / CALL_DAY_STEP)


def previous_runs_url(
    town: Town,
    start: datetime,
    end: datetime,
    *,
    models: Sequence[str] = PREVIOUS_RUN_MODELS,
    variables: Sequence[tuple[str, str]] = PREVIOUS_RUN_VARIABLES,
) -> str:
    """Request archived runs at a fixed lead for every variable and model in one town."""
    hourly = ",".join(
        f"{variable}_previous_day{PREVIOUS_RUN_LEAD_DAYS}" for variable, _unit in variables
    )
    query = urlencode(
        {
            "latitude": town.lat,
            "longitude": town.lon,
            "models": ",".join(models),
            "hourly": hourly,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "timezone": "GMT",
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        }
    )
    return f"{PREVIOUS_RUNS_URL}?{query}"


def previous_run_key(variable: str, model: str, models: Sequence[str]) -> str:
    """Open-Meteo drops the model suffix when the request names a single model."""
    stem = f"{variable}_previous_day{PREVIOUS_RUN_LEAD_DAYS}"
    return stem if len(models) == 1 else f"{stem}_{model}"


def _previous_run_times(payload: PreviousRunsPayload) -> list[datetime]:
    times: list[datetime] = []
    for stamp in payload.hourly.get("time", []):
        if not isinstance(stamp, str):
            msg = "Previous Runs response must carry ISO timestamps"
            raise TypeError(msg)
        times.append(datetime.fromisoformat(stamp).replace(tzinfo=UTC))
    return times


def parse_previous_runs(
    body: bytes,
    town: Town,
    *,
    models: Sequence[str] = PREVIOUS_RUN_MODELS,
) -> list[PreviousRunRow]:
    """Validate units and retain the six-hourly values an observation can be paired with."""
    payload = PreviousRunsPayload.model_validate_json(body)
    if payload.utc_offset_seconds != 0:
        msg = "Previous Runs response must use UTC"
        raise ValueError(msg)
    times = _previous_run_times(payload)
    grid = [(index, valid) for index, valid in enumerate(times) if valid.hour in SCORING_HOURS]
    rows: list[PreviousRunRow] = []
    for model in models:
        for variable, expected_unit in PREVIOUS_RUN_VARIABLES:
            key = previous_run_key(variable, model, models)
            unit = payload.hourly_units.get(key)
            if unit is None or unit == "undefined":
                continue
            if unit != expected_unit:
                msg = f"{key}: expected {expected_unit}, got {unit!r}"
                raise UnitMismatchError(msg)
            numbers = payload.hourly.get(key, [])
            if len(numbers) != len(times):
                msg = f"{key}: incomplete Previous Runs series"
                raise ValueError(msg)
            for index, valid in grid:
                value = numbers[index]
                if isinstance(value, (int, float)) and math.isfinite(value):
                    rows.append(
                        PreviousRunRow(
                            town=town.icao,
                            model=model,
                            variable=variable,
                            valid_time=valid,
                            lead_days=PREVIOUS_RUN_LEAD_DAYS,
                            value=float(value),
                        )
                    )
    return rows


def nearest_observations(
    observations: Sequence[Observation],
    valid_times: Sequence[datetime],
    *,
    window: timedelta = OBSERVATION_WINDOW,
) -> list[Observation]:
    """Pair every six-hourly valid time with the closest report inside the window."""
    series: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    for row in observations:
        series.setdefault((row.town, row.variable), []).append((row.valid_time, row.value))
    aligned: list[Observation] = []
    for (town, variable), points in series.items():
        points.sort()
        stamps = [stamp for stamp, _value in points]
        for target in valid_times:
            index = bisect_left(stamps, target)
            candidates = [choice for choice in (index - 1, index) if 0 <= choice < len(stamps)]
            if not candidates:
                continue
            nearest = min(
                candidates, key=lambda choice: (abs(stamps[choice] - target), stamps[choice])
            )
            if abs(stamps[nearest] - target) <= window:
                aligned.append(Observation(town, variable, target, points[nearest][1]))
    return sorted(aligned, key=lambda row: (row.town, row.variable, row.valid_time))


def request_bytes(url: str) -> bytes:
    """Read a public endpoint with a timeout and three attempts for transient errors."""
    if urlsplit(url).scheme != "https":
        msg = "public data requests require HTTPS"
        raise ValueError(msg)
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            with open_https(url, HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except (URLError, TimeoutError) as error:
            if (
                isinstance(error, HTTPError)
                and error.code < HTTPStatus.INTERNAL_SERVER_ERROR
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
            ) or attempt == len(RETRY_DELAYS_SECONDS):
                msg = f"could not reach {url} ({error})"
                raise UserError(msg, "retry the same collection command", 3) from error
            sleep(RETRY_DELAYS_SECONDS[attempt])
    msg = "HTTP retry loop exhausted"
    raise RuntimeError(msg)


def _write_parquet(
    rows: Sequence[BaseModel | Observation], destination: Path, columns: tuple[str, ...]
) -> None:
    """Write the frame beside its destination first, so a failed run leaves no partial chunk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    records = [row.model_dump() if isinstance(row, BaseModel) else asdict(row) for row in rows]
    pd.DataFrame(records, columns=list(columns)).to_parquet(temporary, index=False)
    temporary.replace(destination)


def collect_wn2(
    towns: Sequence[Town],
    output: Path,
    *,
    now: datetime,
    request: Callable[[str], bytes] = request_bytes,
) -> Path:
    """Write one complete run atomically, rejecting metadata changes during collection."""
    metadata = RunMetadata.model_validate_json(request(METADATA_URL))
    available = datetime.fromtimestamp(metadata.last_run_availability_time, UTC)
    if now < available + REPLICATION_DELAY:
        msg = "WN2 run is still replicating; retry ten minutes after advertised availability"
        retry_at = (available + REPLICATION_DELAY).strftime("%Y-%m-%d %H:%M UTC")
        raise RunNotReadyError(
            msg, f"retry after {retry_at}; see docs/scores.md for metadata checks"
        )
    init = metadata.init_time
    if now - init > timedelta(hours=24) or init > now:
        msg = f"WN2 metadata is stale or future-dated: {init.isoformat()}"
        raise RunNotReadyError(
            msg, "retry this command in 30 minutes; see docs/scores.md for metadata checks"
        )
    rows = [row for town in towns for row in parse_wn2(request(wn2_url(town, init)), town, init)]
    latest = RunMetadata.model_validate_json(request(METADATA_URL))
    if latest.last_run_initialisation_time != metadata.last_run_initialisation_time:
        msg = "WN2 initialization changed during collection; retry the whole run"
        raise RunNotReadyError(
            msg, "retry this command in 10 minutes; see docs/scores.md for metadata checks"
        )
    destination = output / "wn2" / f"{init:%Y-%m-%d-%H}.parquet"
    _write_parquet(rows, destination, FORECAST_COLUMNS)
    return destination


def backfill_plan(towns: Sequence[Town], *, since: datetime, until: datetime) -> list[BackfillDay]:
    """Split the archive into daily chunks, each priced as one town's requests times the towns."""
    days: list[BackfillDay] = []
    start = since
    while start < until:
        end = min(start + timedelta(days=CHUNK_DAYS), until)
        days.append(
            BackfillDay(
                start=start.date(),
                end=(end - timedelta(days=1)).date(),
                calls=len(towns)
                * weighted_calls(
                    len(PREVIOUS_RUN_MODELS) * len(PREVIOUS_RUN_VARIABLES), (end - start).days
                ),
            )
        )
        start = end
    return days


class CallMeter:
    """Count weighted calls and refuse to pass the daily limit part-way through a chunk."""

    def __init__(self, limit: float = DAILY_CALL_LIMIT) -> None:
        self.limit = limit
        self.spent = 0.0

    def add(self, calls: float) -> None:
        """Reserve one request's calls, or stop before the run passes the daily limit."""
        if self.spent + calls > self.limit:
            msg = f"Open-Meteo weighted calls would pass {self.limit:.0f} in one day"
            raise UserError(msg, "resume the backfill on the next scheduled run", 3)
        self.spent += calls


def _chunk_times(day: BackfillDay) -> tuple[datetime, datetime]:
    start = datetime.combine(day.start, time.min, tzinfo=UTC)
    end = datetime.combine(day.end + timedelta(days=1), time.min, tzinfo=UTC)
    return start, end


def _chunk_paths(output: Path, day: BackfillDay) -> tuple[Path, Path]:
    stamp = f"{day.start:%Y-%m-%d}"
    previous_runs = output / "previous_runs" / f"{stamp}.parquet"
    observations = output / "observations" / f"{stamp}.parquet"
    return previous_runs, observations


def _six_hourly(start: datetime, end: datetime) -> list[datetime]:
    times: list[datetime] = []
    cursor = start
    while cursor < end:
        if cursor.hour in SCORING_HOURS:
            times.append(cursor)
        cursor += timedelta(hours=1)
    return times


def collect_backfill(  # noqa: PLR0913 - the archive window and injectable transport are separate
    towns: Sequence[Town],
    output: Path,
    *,
    since: datetime,
    until: datetime,
    request: Callable[[str], bytes] = request_bytes,
    limit: float = DAILY_CALL_LIMIT,
) -> BackfillDay | None:
    """Collect the first chunk that is not stored yet; one chunk per scheduled run."""
    for day in backfill_plan(towns, since=since, until=until):
        previous_runs, observations = _chunk_paths(output, day)
        if previous_runs.exists() and observations.exists():
            continue
        if day.calls > limit:
            msg = f"{day.start} needs {day.calls:.1f} weighted calls, over the {limit:.0f} limit"
            raise UserError(msg, "plan a smaller archive window", 3)
        _collect_chunk(towns, day, output, request=request, limit=limit)
        return day
    return None


def _collect_chunk(
    towns: Sequence[Town],
    day: BackfillDay,
    output: Path,
    *,
    request: Callable[[str], bytes],
    limit: float,
) -> None:
    # The ASOS request reaches back one day so the first 00 UTC slot can see 23:53.
    start, end = _chunk_times(day)
    meter = CallMeter(limit)
    per_request = weighted_calls(
        len(PREVIOUS_RUN_MODELS) * len(PREVIOUS_RUN_VARIABLES), (end - start).days
    )
    runs: list[PreviousRunRow] = []
    observations: list[Observation] = []
    from stormfork.scores.inputs import asos_url, parse_asos  # noqa: PLC0415 - avoids an import cycle

    for town in towns:
        meter.add(per_request)
        runs.extend(parse_previous_runs(request(previous_runs_url(town, start, end)), town))
        observations.extend(
            parse_asos(
                request(asos_url(town, start - timedelta(days=1), end + timedelta(days=1))),
                town,
            )
        )
    aligned = nearest_observations(observations, _six_hourly(start, end))
    previous_runs_path, observations_path = _chunk_paths(output, day)
    _write_parquet(runs, previous_runs_path, PREVIOUS_RUN_COLUMNS)
    _write_parquet(aligned, observations_path, OBSERVATION_COLUMNS)


def probe_wn2_previous_runs(
    town: Town, *, now: datetime, request: Callable[[str], bytes] = request_bytes
) -> bool:
    """Report whether WN2's Previous Runs archive holds any value this week."""
    end = now.date()
    start = end - timedelta(days=WN2_PROBE_DAYS)
    url = previous_runs_url(
        town,
        datetime.combine(start, time.min, tzinfo=UTC),
        datetime.combine(end, time.min, tzinfo=UTC),
        models=("google_weathernext2_ensemble",),
        variables=(("temperature_2m", "°C"),),
    )
    payload = PreviousRunsPayload.model_validate_json(request(url))
    return any(
        isinstance(value, (int, float)) for values in payload.hourly.values() for value in values
    )
