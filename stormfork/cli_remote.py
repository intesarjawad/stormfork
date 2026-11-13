"""Fetch a published experiment when the repository is not on disk.

A release lock in the repository pins one zip per experiment by release tag and
SHA-256. A checkout's own lock is read before the one on the main branch, and
nothing is unpacked until the archive hash matches the lock. Downloads go
through the same transport and offline switch as the model inputs.
"""

import hashlib
import sys
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from pydantic import Field

from stormfork.downloads import https_transfer, require_online
from stormfork.errors import UserError
from stormfork.export.archive import safe_extract
from stormfork.export.divergence import EvidenceModel
from stormfork.export.manifest import Index, Sha256

LOCK_URL = "https://raw.githubusercontent.com/intesarjawad/stormfork/main/site/atlases.lock.json"
RELEASE_URL = "https://github.com/intesarjawad/stormfork/releases/download"
LOCK_FILE = Path("site/atlases.lock.json")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
Slug = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ExperimentId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class Hero(EvidenceModel):
    """The experiment the site opens with and the step where its storm makes landfall."""

    id: ExperimentId
    landfall_step: Index | None = None


class LockEntry(EvidenceModel):
    """One experiment's zip in the pinned release."""

    zip: Slug
    sha256: Sha256


class Lock(EvidenceModel):
    """Release tag and archive hashes for every published experiment."""

    hero: Hero
    release: Slug
    experiments: dict[ExperimentId, LockEntry]


def fetch_bytes(url: str, *, not_found: UserError | None = None) -> bytes:
    """Read a URL, reporting it on stderr; plain HTTP is allowed for loopback only.

    Raises:
        ValueError: The URL is neither HTTPS nor loopback HTTP.
        UserError: Downloads are off, the server answers not_found's 404, or
            the server cannot be reached.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" and not (
        parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS
    ):
        msg = f"downloads require HTTPS: {url}"
        raise ValueError(msg)
    require_online(url)
    sys.stderr.write(f"fetch {url}\n")
    try:
        return b"".join(https_transfer(url).chunks)
    except (URLError, TimeoutError) as error:
        missing = isinstance(error, HTTPError) and error.code == HTTPStatus.NOT_FOUND
        if missing and not_found is not None:
            raise not_found from error
        msg = f"could not reach {url} ({error})"
        raise UserError(msg, "check the connection and retry", 3) from error


def fetch_lock() -> Lock:
    """Read the release lock from the checkout, else download it from the main branch.

    Raises:
        UserError: Downloads are off, no lock is published yet, or the server
            cannot be reached.
    """
    if LOCK_FILE.is_file():
        sys.stderr.write(f"lock {LOCK_FILE}\n")
        return Lock.model_validate_json(LOCK_FILE.read_bytes())
    unpublished = UserError("no published experiments yet", "stormfork scenarios list", 2)
    return Lock.model_validate_json(fetch_bytes(LOCK_URL, not_found=unpublished))


def unpack(data: bytes, destination: Path) -> Path:
    """Extract a zip and return the directory holding its manifest.

    The manifest may sit at the archive root or in a single top-level directory.
    The archive is untrusted, so safe_extract judges it.

    Raises:
        ValueError: The archive is corrupt, unsafe or has no manifest.
    """
    with TemporaryDirectory() as staging:
        archive = Path(staging) / "experiment.zip"
        archive.write_bytes(data)
        safe_extract(archive, destination)
    candidates = [destination, *(path for path in destination.iterdir() if path.is_dir())]
    for directory in candidates:
        if (directory / "manifest.json").is_file():
            return directory
    msg = "archive has no manifest.json at its root or in one top-level directory"
    raise ValueError(msg)


def download_experiment(lock: Lock, experiment: str, destination: Path) -> Path:
    """Fetch an experiment's zip from the pinned release, check its hash and unpack it.

    Raises:
        UserError: The lock does not list the experiment.
        ValueError: The archive hash differs from the lock or the archive is unusable.
    """
    entry = lock.experiments.get(experiment)
    if entry is None:
        msg = f'experiment "{experiment}" is not in the release lock'
        raise UserError(msg, "choose an id from stormfork scenarios list", 2)
    data = fetch_bytes(f"{RELEASE_URL}/{lock.release}/{entry.zip}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != entry.sha256:
        msg = f"{entry.zip} SHA-256 {actual} does not match the lock {entry.sha256}"
        raise ValueError(msg)
    return unpack(data, destination)
