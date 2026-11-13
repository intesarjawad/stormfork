"""The observed track a hindcast is published beside.

Positions, pressure and wind come from the IBTrACS USA columns, so they share
the units and six-hour valid times of the forecast tracks.
"""

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from stormfork.cli_data import parse_time
from stormfork.track.match import TRACK_COLUMNS

STEP_HOURS = 6
PA_PER_MB = 100.0
MS_PER_KNOT = 1852 / 3600
MISSING_VALUE = -9999.0


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number != MISSING_VALUE else None


def observed_track(path: Path, sid: str, init: datetime, steps: int) -> pd.DataFrame:
    """Read one storm's six-hourly IBTrACS samples from init through the last step.

    A sample needs a USA position, pressure and wind; rows lacking any of them,
    the units row and off-cycle rows are left out rather than filled in, so the
    track can end before the last step.

    Raises:
        ValueError: The file lacks the USA columns or has no usable sample.
    """
    if init.tzinfo:
        init = init.astimezone(UTC).replace(tzinfo=None)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        required = {"SID", "ISO_TIME", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            msg = f"IBTrACS file is missing columns: {', '.join(sorted(missing))}"
            raise ValueError(msg)
        samples: dict[int, dict[str, object]] = {}
        for row in reader:
            if (row.get("SID") or "").strip() != sid:
                continue
            try:
                elapsed = parse_time((row.get("ISO_TIME") or "").strip()) - init
            except ValueError:
                continue
            step, remainder = divmod(elapsed, timedelta(hours=STEP_HOURS))
            lat, lon = _number(row.get("USA_LAT")), _number(row.get("USA_LON"))
            wind, pressure = _number(row.get("USA_WIND")), _number(row.get("USA_PRES"))
            if remainder or not 0 <= step <= steps:
                continue
            if lat is None or lon is None or wind is None or pressure is None:
                continue
            samples[step] = {
                "member": 0,
                "step": step,
                "valid_time": init + step * timedelta(hours=STEP_HOURS),
                "lat": lat,
                "lon": lon,
                "mslp_pa": pressure * PA_PER_MB,
                "wind_ms": wind * MS_PER_KNOT,
                "track_id": sid,
            }
    if not samples:
        msg = f"IBTrACS has no usable samples for {sid} from {init.isoformat()}"
        raise ValueError(msg)
    frame = pd.DataFrame([samples[step] for step in sorted(samples)])
    return frame.loc[:, list(TRACK_COLUMNS)]
