"""Write the site's small synthetic fixture, including a no-WebGL frame.

Run from the repository root:
    UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run python -m scripts.export_site_fixture
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scripts.atlas_sizes import synthetic_fields
from stormfork.export.atlas import FRAME_COUNT, pack_atlas
from stormfork.export.divergence import RunName
from stormfork.export.manifest import track_exports
from stormfork.export.preview import write_grayscale
from stormfork.export.quantise import field_metadata, quantise
from stormfork.grid import LAT, LON
from stormfork.track.match import TRACK_COLUMNS

DESTINATION = Path("site/public/experiments/demo-fixture")
FILES = ("mslp", "u10", "v10")
FIELD_NAMES = ("mslp", "wind10")
WIND_RESOLUTION_DEG = 2
FALLBACK_FRAME = 20
TRACK_START = datetime(2026, 1, 1, tzinfo=UTC)
TRACK_STEP_HOURS = 6
TRACK_STEPS = 9


def fork_fields(fields: dict[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
    """Move the baseline a little, smoothly in space and time."""
    phase = np.arange(FRAME_COUNT)[:, None, None] / FRAME_COUNT * 2 * np.pi
    latitude = np.deg2rad(LAT)[None, :, None]
    longitude = np.deg2rad(LON)[None, None, :]
    anomaly = np.cos(latitude) * np.sin(longitude - phase)
    moved = {name: values.copy() for name, values in fields.items()}
    moved["mslp"] += 1200 * anomaly
    moved["u10"] += 6 * anomaly
    moved["v10"] += 4 * anomaly
    return moved


def write_run(
    fields: dict[str, NDArray[np.float64]], destination: Path, run: str
) -> list[tuple[str, str]]:
    """Write one run's atlases and previews, returning relative paths and hashes."""
    written: list[tuple[str, str]] = []
    for file in FILES:
        resolution = WIND_RESOLUTION_DEG if file in ("u10", "v10") else 1
        codes = quantise(fields[file], file)
        atlas = pack_atlas(codes[:, ::resolution, ::resolution], resolution_deg=resolution)
        atlas_path = Path(run) / f"{file}.png"
        preview_path = Path("preview") / f"{run}_{file}_f0.png"
        write_grayscale(atlas, destination / atlas_path)
        write_grayscale(codes[0], destination / preview_path)
        for path in (atlas_path, preview_path):
            digest = hashlib.sha256((destination / path).read_bytes()).hexdigest()
            written.append((path.as_posix(), digest))
    return written


def track_frame(run: RunName, *, end_lat: float, end_lon: float) -> pd.DataFrame:
    """A smooth two-and-a-half-day track for one run."""
    rows: list[dict[str, object]] = []
    for step in range(TRACK_STEPS):
        fraction = step / (TRACK_STEPS - 1)
        rows.append(
            {
                "member": 0,
                "step": step,
                "valid_time": TRACK_START + timedelta(hours=TRACK_STEP_HOURS * step),
                "lat": 15 + (end_lat - 15) * fraction,
                "lon": -60 + (end_lon + 60) * fraction,
                "mslp_pa": 100400 - 900 * fraction,
                "wind_ms": 18 + 12 * fraction,
                "track_id": run,
            }
        )
    return pd.DataFrame(rows, columns=list(TRACK_COLUMNS))


def write_manifest(destination: Path, runs: dict[RunName, list[tuple[str, str]]]) -> None:
    """Write the subset of the exporter manifest the site reads."""
    fields = [
        field
        for field in field_metadata(wind_resolution_deg=WIND_RESOLUTION_DEG)
        if field["name"] in FIELD_NAMES
    ]
    manifest = {
        "id": destination.name,
        "title": "Demo fixture: smooth synthetic fields",
        "fields": fields,
        "runs": {run: dict(assets) for run, assets in runs.items()},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Regenerate the committed fixture in place."""
    fields = synthetic_fields()
    destination = DESTINATION
    runs: dict[RunName, list[tuple[str, str]]] = {
        "baseline": write_run(fields, destination, "baseline"),
        "fork": write_run(fork_fields(fields), destination, "fork"),
    }
    tracks: dict[RunName, pd.DataFrame] = {
        "baseline": track_frame("baseline", end_lat=35, end_lon=-20),
        "fork": track_frame("fork", end_lat=38, end_lon=-14),
    }
    (destination / "tracks.json").write_bytes(track_exports(tracks)["tracks.json"])
    write_grayscale(
        quantise(fields["mslp"], "mslp")[FALLBACK_FRAME],
        destination / "fallback" / "frame20_mslp.png",
    )
    write_manifest(destination, runs)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
