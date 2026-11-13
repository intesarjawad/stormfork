"""Draw the figures in docs/perturbations.md.

The taper profile and the before and after cross-section are computed from
the edit code on a synthetic Gaussian low, so they show the edit and not the
model. The adjustment-ratio histogram and the fork and control separation read
published experiment manifests and are skipped, with a message, for any that
are missing.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import ValidationError

from stormfork.export.divergence import Divergence
from stormfork.export.manifest import Manifest
from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.perturb import Taper, relocate

FIGURES_DIR = Path("docs/figures")
EXPERIMENTS_DIR = Path("site/public/experiments")
CENTRE_LAT = 25.0
CENTRE_LON = 280.0
TAPER_RADIUS_KM = 900.0
LOW_DEPTH_PA = 6000.0
LOW_WIDTH_KM = 130.0
BASE_PRESSURE_PA = 101000.0
EAST_KM = 300.0
SPACING_DEG = 0.25
HOURS_PER_STEP = 6
PROFILE_EXTENT = 1.25  # Radii plotted, so the zero weight beyond the edge is visible.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#555555"


def taper_profile(
    radius_km: float = TAPER_RADIUS_KM,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distance from the taper centre in km and the weight, along the equator eastward."""
    lat = np.arange(-5, 5 + SPACING_DEG, SPACING_DEG)
    grid = Grid.from_dataset(xr.Dataset(coords={"lat": lat, "lon": np.arange(0, 360, SPACING_DEG)}))
    weights = Taper(0.0, 0.0, radius_km).weights(grid)
    row = int(np.argmin(np.abs(grid.lat)))
    east = grid.lon <= np.rad2deg(PROFILE_EXTENT * radius_km / EARTH_RADIUS_KM)
    return great_circle_km(0.0, 0.0, 0.0, grid.lon[east]), weights[row, east]


def synthetic_low() -> xr.Dataset:
    """A Gaussian pressure low on a 0.25 degree grid, named as the edit expects."""
    lat = np.arange(CENTRE_LAT - 15, CENTRE_LAT + 15 + SPACING_DEG, SPACING_DEG)
    lon = np.arange(0, 360, SPACING_DEG)
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    distance = great_circle_km(CENTRE_LAT, CENTRE_LON, lat[:, None], lon)
    shape = np.exp(-0.5 * (distance / LOW_WIDTH_KM) ** 2)
    ds["mean_sea_level_pressure"] = (("lat", "lon"), BASE_PRESSURE_PA - LOW_DEPTH_PA * shape)
    return ds


def relocation_section() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Longitude, then MSLP in hPa before and after the move, on the centre latitude row."""
    before = synthetic_low()
    after = relocate(before, Taper(CENTRE_LAT, CENTRE_LON, TAPER_RADIUS_KM), east_km=EAST_KM)
    window = (before.lon >= CENTRE_LON - 10) & (before.lon <= CENTRE_LON + 10)
    row = int(np.argmin(np.abs(before.lat.values - CENTRE_LAT)))
    pick = np.flatnonzero(window.values)
    return (
        before.lon.values[pick],
        before["mean_sea_level_pressure"].values[row, pick] / 100,
        after["mean_sea_level_pressure"].values[row, pick] / 100,
    )


def load_published(directory: Path) -> list[tuple[Manifest, Divergence]]:
    """Read every experiment directory that holds a full manifest and its divergence."""
    published: list[tuple[Manifest, Divergence]] = []
    for path in sorted(directory.glob("*/manifest.json")) if directory.is_dir() else []:
        try:
            manifest = Manifest.model_validate_json(path.read_bytes())
            divergence = Divergence.model_validate_json(
                (path.parent / manifest.divergence.path).read_bytes()
            )
        except (ValidationError, OSError) as error:
            print(f"skipped {path.parent.name}: {error.__class__.__name__}", file=sys.stderr)
            continue
        published.append((manifest, divergence))
    return published


def save_taper_profile(path: Path) -> None:
    """Weight against distance, with the flat inner half and the cosine falloff."""
    distance, weight = taper_profile()
    figure = Figure(figsize=(6, 3.2), layout="constrained")
    axes = figure.subplots()
    axes.plot(distance, weight, color=BLUE, linewidth=2)
    axes.axvline(TAPER_RADIUS_KM / 2, color=GREY, linestyle=":", label="Half radius")
    axes.axvline(TAPER_RADIUS_KM, color=GREY, linestyle="--", label="Radius")
    axes.set_xlabel("Distance from the taper centre (km)")
    axes.set_ylabel("Edit weight")
    axes.set_title(f"Taper with a {TAPER_RADIUS_KM:g} km radius")
    axes.legend()
    figure.savefig(path, dpi=120)


def save_relocation_section(path: Path) -> None:
    """MSLP along the centre latitude before and after a move east, on a synthetic low."""
    lon, before, after = relocation_section()
    figure = Figure(figsize=(6, 3.2), layout="constrained")
    axes = figure.subplots()
    axes.plot(lon, before, color=BLUE, linewidth=2, label="Before")
    axes.plot(
        lon, after, color=ORANGE, linewidth=2, linestyle="--", label=f"After {EAST_KM:g} km east"
    )
    axes.set_xlabel("Longitude (degrees east)")
    axes.set_ylabel("MSLP (hPa)")
    axes.set_title("Synthetic Gaussian low, 25°N cross-section")
    axes.legend()
    figure.savefig(path, dpi=120)


def save_adjustment_ratios(path: Path, ratios: Sequence[float]) -> None:
    """Histogram of the published experiments' adjustment ratios."""
    figure = Figure(figsize=(6, 3.2), layout="constrained")
    axes = figure.subplots()
    axes.hist(ratios, bins="auto", color=BLUE, edgecolor="black")
    axes.set_xlabel("Adjustment ratio (fork over baseline)")
    axes.set_ylabel("Experiments")
    figure.savefig(path, dpi=120)


def save_separation(path: Path, published: Sequence[tuple[Manifest, Divergence]]) -> None:
    """Fork and control separation from the baseline track, one pair of lines per experiment."""
    figure = Figure(figsize=(6, 3.6), layout="constrained")
    axes = figure.subplots()
    for manifest, divergence in published:
        rows = [
            row
            for row in divergence.steps
            if row.fork_sep_km is not None and row.control_sep_km is not None
        ]
        hours = [row.step * HOURS_PER_STEP for row in rows]
        fork = [row.fork_sep_km for row in rows if row.fork_sep_km is not None]
        control = [row.control_sep_km for row in rows if row.control_sep_km is not None]
        axes.plot(hours, fork, color=BLUE, label=f"{manifest.id} fork")
        axes.plot(hours, control, color=ORANGE, linestyle="--", label=f"{manifest.id} control")
    axes.set_xlabel("Lead time (hours)")
    axes.set_ylabel("Separation from the baseline mean track (km)")
    axes.legend(fontsize="small")
    figure.savefig(path, dpi=120)


def main(argv: Sequence[str] | None = None) -> int:
    """Write every figure whose input exists and name the ones that are skipped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="figure directory")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR, help="experiments")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    save_taper_profile(arguments.out / "taper_profile.png")
    save_relocation_section(arguments.out / "relocation_section.png")
    published = load_published(arguments.experiments)
    ratios = [
        manifest.diagnostics.adjustment_ratio
        for manifest, _ in published
        if manifest.diagnostics.adjustment_ratio is not None
    ]
    if ratios:
        save_adjustment_ratios(arguments.out / "adjustment_ratio.png", ratios)
    else:
        print("adjustment_ratio.png skipped: no published adjustment ratio", file=sys.stderr)
    forks = [item for item in published if item[0].kind != "hindcast"]
    if forks:
        save_separation(arguments.out / "fork_control.png", forks)
    else:
        print("fork_control.png skipped: no published fork experiment", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
