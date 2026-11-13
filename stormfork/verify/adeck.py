"""Read NHC ATCF a-deck track guidance.

An a-deck records each objective aid's forecast positions by model cycle and
forecast lead. Positions encode tenths of a degree with a hemisphere suffix,
for example ``205N`` for 20.5 degrees north and ``947W`` for 94.7 degrees west.
"""

import gzip
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from stormfork.scores.fetch import request_bytes

ADECK_BASE_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public"
_STORM_ID = re.compile(r"^[A-Z]{2}\d{2}\d{4}$")
_COORDINATE = re.compile(r"^([+-]?\d+(?:\.\d+)?)([NSEW])$", re.IGNORECASE)
_TENTHS_PER_DEGREE = 10.0
_MINIMUM_FIELDS = 8
_STORM_FIELD = 0
_NUMBER_FIELD = 1
_CYCLE_FIELD = 2
_AID_FIELD = 4
_LEAD_FIELD = 5
_LATITUDE_FIELD = 6
_LONGITUDE_FIELD = 7
_MAX_LATITUDE_DEG = 90.0
_MAX_LONGITUDE_DEG = 180.0
_BASIN_LENGTH = 2
_CYCLE_LENGTH = 10


@dataclass(frozen=True)
class ADeckPoint:
    """One aid position for a storm, model cycle and forecast lead."""

    storm: str
    aid: str
    init: datetime
    lead_hours: int
    lat: float
    lon: float

    @property
    def valid_time(self) -> datetime:
        """Return the forecast valid time."""
        return self.init + timedelta(hours=self.lead_hours)


@dataclass(frozen=True)
class ADeck:
    """All a-deck positions for one storm, deduplicated by aid, cycle and lead."""

    storm: str
    points: tuple[ADeckPoint, ...]

    def track(self, aid: str, init: datetime) -> dict[int, ADeckPoint]:
        """Return one aid's positions for a model cycle, keyed by lead hours."""
        return {
            point.lead_hours: point
            for point in self.points
            if point.aid == aid and point.init == init
        }

    def aids(self) -> tuple[str, ...]:
        """Return the aid identifiers present in the deck, sorted."""
        return tuple(sorted({point.aid for point in self.points}))

    def inits(self) -> tuple[datetime, ...]:
        """Return the model cycles present in the deck, sorted."""
        return tuple(sorted({point.init for point in self.points}))


def _coordinate(value: str, *, latitude: bool) -> float:
    match = _COORDINATE.fullmatch(value.strip())
    if match is None:
        kind = "latitude" if latitude else "longitude"
        msg = f"invalid {kind} value: {value!r}"
        raise ValueError(msg)
    raw = match.group(1)
    magnitude = float(raw)
    if "." not in raw:
        magnitude /= _TENTHS_PER_DEGREE
    hemisphere = match.group(2).upper()
    if latitude and hemisphere not in {"N", "S"}:
        msg = f"latitude must use N or S: {value!r}"
        raise ValueError(msg)
    if not latitude and hemisphere not in {"E", "W"}:
        msg = f"longitude must use E or W: {value!r}"
        raise ValueError(msg)
    signed = magnitude if hemisphere in {"N", "E"} else -magnitude
    limit = _MAX_LATITUDE_DEG if latitude else _MAX_LONGITUDE_DEG
    if abs(signed) > limit:
        kind = "latitude" if latitude else "longitude"
        msg = f"{kind} outside [-{limit:g}, {limit:g}]: {value!r}"
        raise ValueError(msg)
    return signed


def _fields(line: str) -> list[str]:
    return line.split(",") if "," in line else line.split()


def _point(fields: list[str], line_number: int) -> ADeckPoint:
    if len(fields) < _MINIMUM_FIELDS:
        msg = f"a-deck line {line_number} has {len(fields)} fields; expected at least 8"
        raise ValueError(msg)
    basin = fields[_STORM_FIELD].strip().upper()
    cycle = fields[_CYCLE_FIELD].strip()
    if len(cycle) != _CYCLE_LENGTH or not cycle.isdigit():
        msg = f"a-deck line {line_number} has an invalid cycle: {cycle!r}"
        raise ValueError(msg)
    try:
        number = int(fields[_NUMBER_FIELD].strip())
        init = datetime.fromisoformat(f"{cycle[:4]}-{cycle[4:6]}-{cycle[6:8]}T{cycle[8:10]}:00:00")
        lead = int(fields[_LEAD_FIELD].strip())
    except ValueError as error:
        msg = f"a-deck line {line_number} has an invalid number, cycle or lead"
        raise ValueError(msg) from error
    if len(basin) != _BASIN_LENGTH or not basin.isalpha():
        msg = f"a-deck line {line_number} has an invalid basin: {basin!r}"
        raise ValueError(msg)
    aid = fields[_AID_FIELD].strip().upper()
    if not aid:
        msg = f"a-deck line {line_number} has no aid identifier"
        raise ValueError(msg)
    return ADeckPoint(
        storm=f"{basin}{number:02d}{init.year}",
        aid=aid,
        init=init,
        lead_hours=lead,
        lat=_coordinate(fields[_LATITUDE_FIELD], latitude=True),
        lon=_coordinate(fields[_LONGITUDE_FIELD], latitude=False),
    )


def _deduplicate(points: list[ADeckPoint]) -> tuple[ADeckPoint, ...]:
    unique: dict[tuple[str, datetime, int], ADeckPoint] = {}
    for point in points:
        key = (point.aid, point.init, point.lead_hours)
        previous = unique.get(key)
        if previous is not None and (previous.lat, previous.lon) != (point.lat, point.lon):
            msg = (
                f"{point.aid} at {point.init:%Y-%m-%d %H}Z has conflicting lead {point.lead_hours}"
            )
            raise ValueError(msg)
        unique[key] = point
    return tuple(sorted(unique.values(), key=lambda p: (p.init, p.aid, p.lead_hours)))


def parse_adeck(text: str) -> ADeck:
    """Parse an a-deck with comma or whitespace field separators.

    Blank lines and comment lines are ignored. Repeated rows for different wind
    radius thresholds describe the same position and collapse to one point.

    Raises:
        ValueError: A record is malformed, inconsistent or absent.
    """
    points = [
        _point(_fields(line), line_number)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not points:
        msg = "a-deck contains no forecast records"
        raise ValueError(msg)
    storms = {point.storm for point in points}
    if len(storms) != 1:
        msg = f"a-deck contains multiple storms: {', '.join(sorted(storms))}"
        raise ValueError(msg)
    return ADeck(storm=storms.pop(), points=_deduplicate(points))


def read_adeck(path: str | Path) -> ADeck:
    """Read a plain or gzip-compressed a-deck from a local path."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
            return parse_adeck(stream.read())
    return parse_adeck(path.read_text(encoding="utf-8-sig"))


def adeck_url(storm: str) -> str:
    """Return the public a-deck URL for the current NHC storm identifier.

    Raises:
        ValueError: The identifier is not an ATCF storm identifier.
    """
    identifier = storm.strip().upper()
    if _STORM_ID.fullmatch(identifier) is None:
        msg = f"invalid ATCF storm identifier: {storm!r}"
        raise ValueError(msg)
    return f"{ADECK_BASE_URL}/a{identifier.lower()}.dat.gz"


def fetch_adeck(storm: str, request: Callable[[str], bytes] = request_bytes) -> ADeck:
    """Download and parse the current season's public a-deck.

    Raises:
        ValueError: The response is not gzip data or belongs to another storm.
    """
    identifier = storm.strip().upper()
    body = gzip.decompress(request(adeck_url(identifier)))
    deck = parse_adeck(body.decode("utf-8-sig"))
    if deck.storm != identifier:
        msg = f"a-deck URL for {identifier} returned {deck.storm}"
        raise ValueError(msg)
    return deck
