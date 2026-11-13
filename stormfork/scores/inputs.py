"""Fetch the observations and IFS forecasts the scoreboard compares."""

import csv
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field

from stormfork.scores.build import IFS_COLUMNS, OBSERVATION_COLUMNS
from stormfork.scores.fetch import nearest_observations, request_bytes
from stormfork.scores.metrics import IfsForecast, Observation
from stormfork.scores.towns import Town

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IFS_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
IFS_MODEL = "ecmwf_ifs025"
ASOS_VARIABLES = (
    ("tmpf", "temperature_2m"),
    ("mslp", "pressure_msl"),
    ("sknt", "wind_speed_10m"),
)
VARIABLE_UNITS = (("temperature_2m", "°C"), ("pressure_msl", "hPa"), ("wind_speed_10m", "m/s"))
KNOTS_TO_MS = 0.514444
FAHRENHEIT_OFFSET = 32.0
FAHRENHEIT_SCALE = 5 / 9
MISSING = {"", "M", "T"}


class ForecastPayload(BaseModel):
    """The JSON fields needed to validate a historical forecast response."""

    model_config = ConfigDict(strict=True)
    utc_offset_seconds: int = Field(strict=True)
    hourly_units: dict[str, str]
    hourly: dict[str, list[float | str | None]]


def asos_url(town: Town, start: datetime, end: datetime) -> str:
    """Request the three station variables for the scoring window in UTC."""
    pairs: list[tuple[str, str]] = [
        ("station", town.icao),
        *[("data", name) for name, _variable in ASOS_VARIABLES],
        ("year1", str(start.year)),
        ("month1", str(start.month)),
        ("day1", str(start.day)),
        ("year2", str(end.year)),
        ("month2", str(end.month)),
        ("day2", str(end.day)),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
    ]
    return f"{ASOS_URL}?{urlencode(pairs)}"


def ifs_url(town: Town, start: datetime, end: datetime) -> str:
    """Request the IFS analysis-to-forecast series for the scoring window."""
    query = urlencode(
        {
            "latitude": town.lat,
            "longitude": town.lon,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(variable for variable, _unit in VARIABLE_UNITS),
            "models": IFS_MODEL,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "timezone": "GMT",
        }
    )
    return f"{IFS_URL}?{query}"


def _number(value: str | None) -> float | None:
    if value is None or value.strip() in MISSING:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_asos(body: bytes, town: Town) -> list[Observation]:
    """Convert IEM's Fahrenheit, hPa and knot columns to the scoreboard units."""
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    rows: list[Observation] = []
    for row in reader:
        valid = row.get("valid")
        if valid is None:
            continue
        valid_time = datetime.fromisoformat(valid).replace(tzinfo=UTC)
        for name, variable in ASOS_VARIABLES:
            value = _number(row.get(name))
            if value is None:
                continue
            if variable == "temperature_2m":
                value = (value - FAHRENHEIT_OFFSET) * FAHRENHEIT_SCALE
            elif variable == "wind_speed_10m":
                value *= KNOTS_TO_MS
            rows.append(Observation(town.icao, variable, valid_time, value))
    return rows


def parse_ifs(body: bytes, town: Town) -> list[IfsForecast]:
    """Read the historical forecast series as lead-zero IFS comparisons."""
    payload = ForecastPayload.model_validate_json(body)
    if payload.utc_offset_seconds != 0:
        msg = "IFS response must use UTC"
        raise ValueError(msg)
    times = payload.hourly.get("time", [])
    rows: list[IfsForecast] = []
    for variable, expected_unit in VARIABLE_UNITS:
        if payload.hourly_units.get(variable) != expected_unit:
            msg = (
                f"{variable}: expected {expected_unit}, got {payload.hourly_units.get(variable)!r}"
            )
            raise ValueError(msg)
        values = payload.hourly.get(variable, [])
        if len(values) != len(times):
            msg = f"{variable}: incomplete IFS horizon"
            raise ValueError(msg)
        for stamp, value in zip(times, values, strict=True):
            if not isinstance(stamp, str) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            valid_time = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
            rows.append(IfsForecast(town.icao, variable, valid_time, 0, float(value)))
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_inputs(
    towns: Sequence[Town],
    *,
    start: datetime,
    end: datetime,
    output: Path,
    request: Callable[[str], bytes] = request_bytes,
) -> tuple[Path, Path]:
    """Fetch every town and write the two CSVs the build command reads."""
    observations: list[Observation] = []
    ifs: list[IfsForecast] = []
    for town in towns:
        reports = parse_asos(
            request(asos_url(town, start - timedelta(days=1), end + timedelta(days=1))), town
        )
        forecasts = [
            row
            for row in parse_ifs(request(ifs_url(town, start, end)), town)
            if start <= row.valid_time < end
        ]
        targets = sorted({row.valid_time for row in forecasts})
        observations.extend(nearest_observations(reports, targets))
        ifs.extend(forecasts)
    output.mkdir(parents=True, exist_ok=True)
    observations_path = output / "observations.csv"
    ifs_path = output / "ifs.csv"
    _write_csv(observations_path, OBSERVATION_COLUMNS, [asdict(row) for row in observations])
    _write_csv(ifs_path, IFS_COLUMNS, [asdict(row) for row in ifs])
    (output / "scoring-day.txt").write_text(f"{end.date().isoformat()}\n", encoding="utf-8")
    return observations_path, ifs_path
