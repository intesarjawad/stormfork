"""Extract the retained Katrina taper row from a local NOAA North Atlantic CSV.

Download SOURCE_URL to /tmp/ibtracs.NA.list.v04r01.csv, then run:
python scripts/extract_ibtracs_taper.py /tmp/ibtracs.NA.list.v04r01.csv \
    tests/fixtures/ibtracs_katrina_taper.csv --retrieved 2026-11-05
"""

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

SOURCE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
SID = "2005236N23285"
ISO_TIME = "2005-08-24 12:00:00"


def extract(source: Path, destination: Path, retrieved: date) -> None:
    """Retain the full matching row and units, with source hashes in a sidecar."""
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        units = next(reader)
        rows = [row for row in reader if row["SID"] == SID and row["ISO_TIME"] == ISO_TIME]
        columns = reader.fieldnames
    if len(rows) != 1 or columns is None:
        msg = f"expected one row for {SID} at {ISO_TIME}, found {len(rows)}"
        raise ValueError(msg)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerow(units)
        writer.writerows(rows)
    metadata = {
        "source_url": SOURCE_URL,
        "version": "v04r01",
        "retrieved": retrieved.isoformat(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "selection": {"SID": SID, "ISO_TIME": ISO_TIME},
        "column_documentation": (
            "https://www.ncei.noaa.gov/sites/default/files/2025-09/"
            "IBTrACS_v04r01_column_documentation.pdf"
        ),
    }
    destination.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Read a downloaded CSV and write the selected row and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--retrieved", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    extract(args.source, args.destination, args.retrieved)


if __name__ == "__main__":
    main()
