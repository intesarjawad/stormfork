"""Check a contributor's experiment archive before it is published, and record the release.

An archive is published only when its files are the ones the merged manifest
names and a second person, not the one who ran it, has reproduced it. The
`check` command refuses otherwise; `lock` pins the archive in the release lock.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from stormfork.cli_remote import Lock, LockEntry, unpack
from stormfork.export.manifest import Manifest, manifest_paths, verify_manifest
from stormfork.export.reproduction import Reproductions

EXPERIMENTS_DIR = Path("site/public/experiments")
LOCK_FILE = Path("site/atlases.lock.json")
MANIFEST = "manifest.json"
LEDGER = "reproductions.json"


def check_reproduction(ledger: Path, manifest: Manifest) -> None:
    """Require a passing reproduction by someone other than whoever ran the experiment.

    The ledger's `by` values were held to their pull request authors when they
    were merged, so they can be trusted here.

    Raises:
        ValueError: The experiment has no runner, no ledger or no independent pass.
    """
    if manifest.run_by is None:
        msg = f"{manifest.id} has no run_by in its manifest"
        raise ValueError(msg)
    if not ledger.is_file():
        msg = f"{manifest.id} has no {LEDGER}"
        raise ValueError(msg)
    entries = Reproductions.validate_json(ledger.read_bytes())
    runner = manifest.run_by.casefold()
    if not any(entry.passed and entry.by.casefold() != runner for entry in entries):
        msg = f"{manifest.id} has no passing reproduction by anyone but {manifest.run_by}"
        raise ValueError(msg)


def check_experiment(
    archive: Path, experiment_id: str, experiments: Path, workdir: Path
) -> Manifest:
    """Unpack an archive safely and check it against the merged manifest and ledger.

    Raises:
        ValueError: The archive is unsafe, differs from the merged manifest, holds
            unlisted files, or the experiment lacks an independent reproduction.
    """
    merged = experiments / experiment_id / MANIFEST
    if not merged.is_file():
        msg = f"{merged} is not merged yet"
        raise ValueError(msg)
    manifest = Manifest.model_validate_json(merged.read_bytes())
    if manifest.id != experiment_id:
        msg = f'{merged} holds "{manifest.id}", not "{experiment_id}"'
        raise ValueError(msg)
    check_reproduction(experiments / experiment_id / LEDGER, manifest)
    directory = unpack(archive.read_bytes(), workdir)
    if (directory / MANIFEST).read_bytes() != merged.read_bytes():
        msg = f"the archive's {MANIFEST} differs from the merged one"
        raise ValueError(msg)
    verify_manifest(directory)
    unlisted = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    } - manifest_paths(manifest)
    if unlisted:
        msg = f"the archive holds files the manifest does not name: {sorted(unlisted)}"
        raise ValueError(msg)
    return manifest


def update_lock(lock_path: Path, experiment_id: str, archive: Path, release: str) -> None:
    """Pin the archive's name and SHA-256 under a new release tag."""
    lock = Lock.model_validate_json(lock_path.read_bytes())
    entry = LockEntry(zip=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    updated = Lock(
        hero=lock.hero, release=release, experiments={**lock.experiments, experiment_id: entry}
    )
    lock_path.write_text(updated.model_dump_json(indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the pre-publication checks or update the release lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check an archive before it is published")
    check.add_argument("experiment_id")
    check.add_argument("archive", type=Path)
    check.add_argument("--workdir", type=Path, required=True, help="empty directory to unpack into")
    check.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    lock = commands.add_parser("lock", help="pin an archive in the release lock")
    lock.add_argument("experiment_id")
    lock.add_argument("archive", type=Path)
    lock.add_argument("--release", required=True, help="release tag, such as atlases-20261107")
    lock.add_argument("--lock", type=Path, default=LOCK_FILE)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            check_experiment(args.archive, args.experiment_id, args.experiments, args.workdir)
            print(f"{args.experiment_id}: archive matches the merged manifest and is reproduced")
        else:
            update_lock(args.lock, args.experiment_id, args.archive, args.release)
            print(f"{args.lock}: {args.experiment_id} pinned in {args.release}")
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
