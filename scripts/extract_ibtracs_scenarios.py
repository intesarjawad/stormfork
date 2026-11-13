"""Extract the IBTrACS rows that the experiment list depends on from a local NOAA CSV.

Download SOURCE_URL to /tmp/ibtracs.NA.list.v04r01.csv, then run:
python scripts/extract_ibtracs_scenarios.py /tmp/ibtracs.NA.list.v04r01.csv \
    tests/fixtures/ibtracs_scenarios.csv --retrieved 2026-11-07

The fixture keeps every landfall record (USA_RECORD "L") of the hindcast storms
and the init-time rows from which fork tapers are derived.
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
LANDFALL_SIDS = ("2024268N17278", "2024279N21265", "2025294N14290")
TAPER_ROWS = (
    ("2012296N14283", "2012-10-26 00:00:00"),
    ("2005236N23285", "2005-08-27 00:00:00"),
    ("2024279N21265", "2024-10-07 00:00:00"),
)


def selected(row: dict[str, str]) -> bool:
    """Return whether a row is a hindcast landfall or a taper init row."""
    landfall = row["SID"] in LANDFALL_SIDS and row["USA_RECORD"].strip() == "L"
    return landfall or (row["SID"], row["ISO_TIME"]) in TAPER_ROWS


def extract(source: Path, destination: Path, retrieved: date) -> None:
    """Retain the selected rows and units, with source hashes in a sidecar."""
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        units = next(reader)
        rows = [row for row in reader if selected(row)]
        columns = reader.fieldnames
    expected = len(TAPER_ROWS) + len(LANDFALL_SIDS)
    if columns is None or len(rows) < expected:
        msg = f"expected at least {expected} rows, found {len(rows)}"
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
        "selection": {
            "landfall_sids": list(LANDFALL_SIDS),
            "taper_rows": [{"SID": sid, "ISO_TIME": time} for sid, time in TAPER_ROWS],
        },
        "column_documentation": (
            "https://www.ncei.noaa.gov/sites/default/files/2025-09/"
            "IBTrACS_v04r01_column_documentation.pdf"
        ),
    }
    destination.with_suffix(".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Read a downloaded CSV and write the selected rows and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--retrieved", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    extract(args.source, args.destination, args.retrieved)


if __name__ == "__main__":
    main()
