"""Check download caps in decimal MB, including previews in each run's cap.

Also measures the gzip size of the built site's JavaScript. Build first, then record:
(cd site && pnpm run build)
uv run python scripts/perf_budget.py --bundle site/dist --record docs/benchmarks.md
The record step replaces only the Perf budget section, so run it after atlas_sizes.py,
which rewrites the whole file.
"""

import argparse
import gzip
import shlex
import sys
from pathlib import Path

KB = 1_000
MB = 1_000_000
BUNDLE_CAP_BYTES = 250 * KB
GZIP_LEVEL = 9
BUILD_COMMAND = "(cd site && pnpm run build)"
SECTION_HEADING = "## Perf budget"
RUN_CAP_BYTES = 8 * MB
EXPERIMENT_CAP_BYTES = 24 * MB
COLLECTION_CAP_BYTES = 250 * MB
HERO_MSLP_CAP_BYTES = 6 * MB
SWEEP_MSLP_CAP_BYTES = 2 * MB
HERO_RUNS = ("baseline", "fork", "control")


def _folder_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def _check_size(label: str, actual: int, cap: int) -> None:
    if actual > cap:
        msg = f"{label}: {actual} bytes exceeds {cap} bytes"
        raise ValueError(msg)


def check_budgets(root: Path, *, hero: str, sweep_stops: tuple[str, ...] = ()) -> None:
    """Check run, experiment, collection, hero MSLP and sweep-stop MSLP sizes.

    Experiments are direct children of root. Run previews live in the sibling
    preview directory and count toward that run as well as the experiment.
    Hero and sweep names refer to experiment directory names; sweep MSLP lives
    in each stop's baseline directory. Required default-layer files must exist.

    Raises:
        ValueError: A cap fails or an experiment name is not a directory basename.
        FileNotFoundError: A required directory or default-layer file is missing.
    """
    for name in (hero, *sweep_stops):
        if not name or name in (".", "..") or any(char in name for char in "/\\:"):
            msg = "experiment names must be single directory names"
            raise ValueError(msg)
    if not root.is_dir():
        raise FileNotFoundError(root)
    _check_size("collection", _folder_bytes(root), COLLECTION_CAP_BYTES)
    for experiment in sorted(path for path in root.iterdir() if path.is_dir()):
        _check_size(experiment.name, _folder_bytes(experiment), EXPERIMENT_CAP_BYTES)
        for run in sorted(path for path in experiment.iterdir() if path.is_dir()):
            if run.name == "preview":
                continue
            previews = sum(
                path.stat().st_size
                for path in (experiment / "preview").glob(f"{run.name}_*_f0.png")
            )
            _check_size(
                f"{experiment.name}/{run.name}", _folder_bytes(run) + previews, RUN_CAP_BYTES
            )
    hero_size = sum((root / hero / run / "mslp.png").stat().st_size for run in HERO_RUNS)
    _check_size(f"{hero} default MSLP", hero_size, HERO_MSLP_CAP_BYTES)
    for stop in sweep_stops:
        size = (root / stop / "baseline" / "mslp.png").stat().st_size
        _check_size(f"{stop} MSLP", size, SWEEP_MSLP_CAP_BYTES)


def bundle_gzip_sizes(dist: Path) -> dict[str, int]:
    """Measure each JavaScript file under dist, gzipped at a fixed level without a timestamp.

    Keys are paths relative to dist, in sorted order.

    Raises:
        FileNotFoundError: dist is missing or holds no JavaScript.
    """
    if not dist.is_dir():
        raise FileNotFoundError(dist)
    files = sorted(path for path in dist.rglob("*") if path.suffix in {".js", ".mjs"})
    if not files:
        msg = f"no JavaScript found in {dist}"
        raise FileNotFoundError(msg)
    return {
        path.relative_to(dist).as_posix(): len(
            gzip.compress(path.read_bytes(), compresslevel=GZIP_LEVEL, mtime=0)
        )
        for path in files
    }


def check_bundle(sizes: dict[str, int]) -> None:
    """Check that all JavaScript together stays within the gzip cap.

    Raises:
        ValueError: The summed gzip size exceeds BUNDLE_CAP_BYTES.
    """
    _check_size("gzip JS bundle", sum(sizes.values()), BUNDLE_CAP_BYTES)


def bundle_section(sizes: dict[str, int], measure_command: str) -> str:
    """Render the measured bundle sizes as a Markdown section, with the commands used."""
    lines = [
        SECTION_HEADING,
        "",
        f"Gzip level {GZIP_LEVEL}, sizes in bytes; KB means 1,000 bytes.",
        "The bundle is every .js and .mjs file in the build output.",
        "",
        "Commands, run from the repository root:",
        "",
        "```",
        BUILD_COMMAND,
        measure_command,
        "```",
        "",
        "| File | Gzip bytes |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {size} |" for name, size in sizes.items())
    lines.extend(
        (
            f"| Total | {sum(sizes.values())} |",
            "",
            f"Cap: {BUNDLE_CAP_BYTES} bytes.",
            "",
        )
    )
    return "\n".join(lines)


def replace_section(text: str, section: str) -> str:
    """Return text with the Perf budget section replaced, or appended when absent."""
    start = text.find(f"\n{SECTION_HEADING}\n")
    if start == -1:
        return f"{text.rstrip()}\n\n{section}"
    end = text.find("\n## ", start + 1)
    tail = "" if end == -1 else text[end:]
    return f"{text[: start + 1]}{section}{tail}"


def main() -> None:
    """Validate the experiment caps and the gzip JS bundle cap, and optionally record the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--hero")
    parser.add_argument("--sweep-stop", action="append", default=[])
    parser.add_argument("--bundle", type=Path, help="built site directory to measure")
    parser.add_argument("--record", type=Path, help="Markdown file to receive the bundle section")
    args = parser.parse_args()
    if args.root is None and args.bundle is None:
        parser.error("give an experiment root, --bundle, or both")
    if args.root is not None and not args.hero:
        parser.error("an experiment root requires --hero")
    if args.record is not None and args.bundle is None:
        parser.error("--record requires --bundle")
    if args.root is not None:
        check_budgets(args.root, hero=args.hero, sweep_stops=tuple(args.sweep_stop))
        print("PASS: run, experiment, collection, hero MSLP and sweep MSLP budgets")
    if args.bundle is not None:
        sizes = bundle_gzip_sizes(args.bundle)
        if args.record is not None:
            command = "uv run python scripts/perf_budget.py " + shlex.join(sys.argv[1:])
            section = bundle_section(sizes, command)
            existing = args.record.read_text(encoding="utf-8") if args.record.exists() else ""
            args.record.write_text(replace_section(existing, section), encoding="utf-8")
        check_bundle(sizes)
        print(f"PASS: gzip JS bundle {sum(sizes.values())} bytes within {BUNDLE_CAP_BYTES} bytes")


if __name__ == "__main__":
    main()
