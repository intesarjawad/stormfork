"""Local file boundaries for command line operations.

I use explicit file inputs until a scenario catalogue is available. A directory
is a local Zarr store; a file is NetCDF. No command resolves remote datasets.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path

import xarray as xr

from stormfork.errors import UserError
from stormfork.perturb import Taper


def parse_time(value: str) -> datetime:
    """Normalize ISO timestamps to naive UTC for dataset and CSV comparisons."""
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed


def load_dataset(path: Path) -> xr.Dataset:
    """Load a local dataset and release its file handles before computation."""
    if not path.exists():
        msg = f'dataset "{path}" does not exist'
        raise UserError(msg, "supply a local NetCDF or Zarr path", 2)
    return xr.load_dataset(path, engine="zarr" if path.is_dir() else None, chunks=None)


def taper_from_csv(path: Path, sid: str, init: str) -> Taper:
    """Select exactly one IBTrACS CSV row, allowing the standard units row."""
    instant = parse_time(init)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True, strict=True)
        rows = [
            row
            for row in reader
            if (row.get("SID") or "").strip() == sid
            and parse_time((row.get("ISO_TIME") or "").strip()) == instant
        ]
    if len(rows) != 1:
        msg = f"{sid} at {init}: expected one IBTrACS row, found {len(rows)}"
        raise UserError(
            msg,
            "select a storm ID and init time present exactly once in the CSV",
            2,
        )
    return Taper.from_ibtracs(rows[0])
