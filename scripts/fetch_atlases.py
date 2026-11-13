"""Fetch the published experiment atlases, or pin a new atlases release in the lock.

Without --lock, every zip that site/atlases.lock.json pins is downloaded from
its GitHub release, checked against the lock's SHA-256 and unpacked into
site/public/experiments/<id>/. That needs network access; with
STORMFORK_OFFLINE set it fetches nothing and fails. With --lock TAG, run after
gh release create uploads dist/atlases/*.zip, the lock is rewritten to pin
those zips, by SHA-256, as the assets of release TAG.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from stormfork.cli_export import ATLASES_DIR
from stormfork.cli_remote import LOCK_FILE, Hero, Lock, LockEntry, download_experiment
from stormfork.errors import UserError
from stormfork.export.manifest import verify_manifest

EXPERIMENTS_DIR = Path("site/public/experiments")
HERO_FILE = Path("site/hero.json")


def write_lock(tag: str) -> str:
    """Pin every zip in the atlases directory as an asset of the release tag.

    Raises:
        ValueError: There is no zip to pin, or the tag is not a release name.
        OSError: The hero record cannot be read.
    """
    archives = sorted(ATLASES_DIR.glob("*.zip"))
    if not archives:
        msg = f"{ATLASES_DIR} holds no zip; export the experiments first"
        raise ValueError(msg)
    lock = Lock(
        hero=Hero.model_validate_json(HERO_FILE.read_bytes()),
        release=tag,
        experiments={
            archive.stem: LockEntry(
                zip=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
            )
            for archive in archives
        },
    )
    LOCK_FILE.write_text(lock.model_dump_json(indent=2) + "\n")
    return f"{LOCK_FILE}: {len(archives)} experiments pinned in {tag}"


def fetch_atlases() -> str:
    """Download, check and unpack every experiment the lock pins.

    Each archive is unpacked into a temporary directory first, so a refused
    archive never touches the files already in the checkout.

    Without a lock nothing is published yet, and the site runs on its test fixtures.

    Raises:
        UserError: Downloads are off or the server cannot be reached.
        ValueError: An archive differs from the lock, is unsafe or holds another experiment.
    """
    if not LOCK_FILE.is_file():
        return "no atlases lock; fixtures only"
    lock = Lock.model_validate_json(LOCK_FILE.read_bytes())
    for experiment in sorted(lock.experiments):
        with TemporaryDirectory() as unpacked:
            directory = download_experiment(lock, experiment, Path(unpacked))
            manifest = verify_manifest(directory)
            if manifest.id != experiment:
                msg = f'the release holds "{manifest.id}" under the id "{experiment}"'
                raise ValueError(msg)
            shutil.copytree(directory, EXPERIMENTS_DIR / experiment, dirs_exist_ok=True)
    return f"{len(lock.experiments)} experiments unpacked into {EXPERIMENTS_DIR}"


def main(argv: list[str] | None = None) -> int:
    """Fetch the pinned atlases, or pin a release with --lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", metavar="TAG", help="release tag, such as atlases-20261107")
    args = parser.parse_args(argv)
    try:
        print(write_lock(args.lock) if args.lock else fetch_atlases())
    except UserError as error:
        print(f"error: {error.what}\nnext: {error.next_step}", file=sys.stderr)
        return error.exit_code
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
