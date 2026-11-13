"""Measure atlas exports in decimal bytes, or generate full-size synthetic fields.

Synthetic: UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run python scripts/atlas_sizes.py \
    --synthetic --output /tmp/stormfork-atlas-sizes --markdown docs/benchmarks.md
Existing exports: uv run python scripts/atlas_sizes.py /path/to/experiment \
    --label 'Forecast export' --markdown /tmp/atlas-sizes.md
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stormfork.export.atlas import FRAME_COUNT, export_run
from stormfork.export.quantise import FILE_NAMES
from stormfork.grid import LAT, LON

WIND_BOX_HALF_WIDTH_DEG = 5


def synthetic_fields() -> dict[str, NDArray[np.float64]]:
    """Build 41 smooth analytic 1° fields, without model data or size predictions."""
    phase = np.arange(FRAME_COUNT)[:, None, None] / FRAME_COUNT * 2 * np.pi
    lat = np.deg2rad(LAT)[None, :, None]
    lon = np.deg2rad(LON)[None, None, :]
    wave = np.cos(lat) * np.sin(lon - phase)
    core = np.exp(-(((LAT[None, :, None] - 20) / 12) ** 2)) * np.exp(
        -((np.sin((lon - phase - np.deg2rad(300)) / 2) / 0.12) ** 2)
    )
    fields = {
        "mslp": 101000 + 1500 * wave - 6500 * core,
        "u10": 12 * wave + 30 * core,
        "v10": -10 * wave - 40 * core,
        "t2m": 260 + 35 * np.cos(lat) + 5 * wave,
        "z500": 53000 + 4000 * wave,
        "tp6h": 0.08 * core + 0.002 * (wave + 1),
    }
    box = (abs(LAT[:, None] - 20) <= WIND_BOX_HALF_WIDTH_DEG) & (
        abs(LON[None, :] - 300) <= WIND_BOX_HALF_WIDTH_DEG
    )
    fields["u10"][0] = np.where(box, 30, 0)
    fields["v10"][0] = np.where(box, -40, 0)
    fields["tp6h"][0] = np.nan
    return fields


def size_table(experiment: Path, label: str) -> str:
    """Measure all six atlases and corresponding previews for each run.

    Raises:
        ValueError: No run folders containing mslp.png exist.
    """
    runs = sorted(path.parent for path in experiment.glob("*/mslp.png"))
    if not runs:
        msg = f"no run atlases found in {experiment}"
        raise ValueError(msg)
    lines = [
        "# Atlas sizes",
        "",
        label,
        "",
        "Sizes are filesystem bytes; MB means 1,000,000 bytes. PNGs use lossless",
        "8-bit grayscale with Pillow optimize=True. Each run contains 41 frames.",
        "",
        "| Run | File | Atlas bytes | Preview bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    for run in runs:
        atlas_total = preview_total = 0
        for file in FILE_NAMES:
            atlas_size = (run / f"{file}.png").stat().st_size
            preview_size = (experiment / "preview" / f"{run.name}_{file}_f0.png").stat().st_size
            atlas_total += atlas_size
            preview_total += preview_size
            lines.append(f"| {run.name} | {file} | {atlas_size} | {preview_size} |")
        lines.append(f"| {run.name} | Total | {atlas_total} | {preview_total} |")
        lines.extend(("", f"Run including previews: {atlas_total + preview_total} bytes.", ""))
    return "\n".join(lines)


def main() -> None:
    """Write a measured size table from synthetic fields or an existing export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wind-resolution-deg", type=int, choices=(1, 2), default=1)
    parser.add_argument("--label")
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic:
        if args.output is None or args.experiment is not None:
            parser.error("--synthetic requires --output and no experiment argument")
        experiment = Path(args.output)
        export_run(
            synthetic_fields(),
            experiment,
            "baseline",
            wind_resolution_deg=args.wind_resolution_deg,
        )
        label = (
            "Synthetic full-size analytic fields on a 181 x 360 grid, 41 frames. "
            f"Wind atlases use {args.wind_resolution_deg}° sampling; all previews are 360 x 181. "
            "These measurements do not estimate model-output compression. "
            "Real-run measurements are pending."
        )
    else:
        if args.experiment is None or not args.label or args.output is not None:
            parser.error("existing exports require experiment and --label, without --output")
        experiment = Path(args.experiment)
        label = str(args.label)
    table = size_table(experiment, label)
    args.markdown.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
