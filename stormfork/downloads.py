"""Model weights, the HRES sample and IBTrACS, from a local cache or their publishers.

Every input is pinned here by URL, and by SHA-256 where the publisher's file
never changes. A download goes to a hidden file beside its destination and is
renamed only once it verifies, so an interrupted or corrupt transfer never
leaves a file that a later command would trust.

How the digests were obtained: the weights (226,897,594 bytes) and the sample
(79,552,583 bytes) were downloaded in full on 2026-11-05. Each copy's base64
MD5 matched the bucket's own ``x-goog-hash`` header, re-read on 2026-11-09
(``R1qGPfvNoqNl62xzJfz6XA==`` and ``KDz7si0OQ8hLxTAQSuyH5Q==``), and the
SHA-256 below was computed from those verified copies. NCEI rewrites the
IBTrACS file in place several times a week (Last-Modified 2026-11-03, then
2026-11-08), so it has no fixed digest: it is pinned by URL, and a transfer
must deliver the length the server announced.
"""

import errno
import hashlib
import os
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, Request, build_opener

from stormfork.errors import UserError

CACHE_ENV = "STORMFORK_CACHE_DIR"
OFFLINE_ENV = "STORMFORK_OFFLINE"
HTTP_TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 20
# Left free after a download so the cache never fills the disk it shares.
DISK_HEADROOM_BYTES = 512 << 20
BYTES_PER_MB = 1_000_000
# A temporary file is private to its owner; the finished download is an ordinary file.
CACHE_FILE_MODE = 0o644
BUCKET = "https://storage.googleapis.com/dm_graphcast/weathernext2"
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-"
    "ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc"
)


@dataclass(frozen=True)
class Source:
    """One downloadable input and the command-line flag that overrides it."""

    flag: str
    title: str
    url: str
    filename: str
    sha256: str | None
    credit: str


WEIGHTS = Source(
    flag="weights",
    title="model weights",
    url=f"{BUCKET}/params/WeatherNextCyclones_Mini_%3C2024.npz",
    filename="WeatherNextCyclones_Mini_2024.npz",
    sha256="a1bb151457077248d70a458b1a7b19deacd2926fbcb46eee4ff2691444e3596e",
    credit="WeatherNext Cyclones, Google DeepMind, CC BY 4.0.",
)
HRES_SAMPLE = Source(
    flag="sample",
    title="HRES sample",
    url=(
        f"{BUCKET}/dataset/"
        "source-hres_forecast_init-2024-10-07%2000:00:00_res-1.0_levels-13_steps-01.nc"
    ),
    filename="hres_2024-10-07_01.nc",
    sha256="ea28e70fc2b5828af14c9057c45c562a3887da2b1cd951472a841da2630a742d",
    # The ECMWF statement the weathernext v0.3.0 README gives for its HRES examples.
    credit=(
        "ECMWF HRES data © 2023 European Centre for Medium-Range Weather Forecasts "
        "(ECMWF), source www.ecmwf.int, CC BY 4.0, as modified by Google."
    ),
)
IBTRACS = Source(
    flag="ibtracs",
    title="IBTrACS",
    url=IBTRACS_URL,
    filename="IBTrACS.ALL.v04r01.nc",
    sha256=None,
    credit="IBTrACS v04r01, NOAA NCEI, public domain (Knapp et al. 2010).",
)
SOURCES: Sequence[Source] = (WEIGHTS, HRES_SAMPLE, IBTRACS)


class Transfer(NamedTuple):
    """An opened download: its announced length, if any, and its body in chunks."""

    size_bytes: int | None
    chunks: Iterator[bytes]


Transport = Callable[[str], Transfer]


def require_online(url: str, environ: Mapping[str, str] = os.environ) -> None:
    """Refuse a network read while downloads are off, naming what it would have fetched.

    Raises:
        UserError: The offline variable is set.
    """
    if is_offline(environ):
        msg = f"{OFFLINE_ENV} is set, so nothing was fetched from {url}"
        raise UserError(msg, f"unset {OFFLINE_ENV} and retry", 2)


def open_https(
    request: str | Request, timeout_seconds: float = HTTP_TIMEOUT_SECONDS
) -> HTTPResponse:
    """Open a connection unless downloads are off; every request the package makes starts here.

    Raises:
        UserError: Downloads are off.
    """
    require_online(request if isinstance(request, str) else request.full_url)
    return build_opener(HTTPSHandler()).open(request, timeout=timeout_seconds)


def https_transfer(url: str) -> Transfer:
    """Open an HTTPS download; reading the chunks to the end closes the connection."""
    response = open_https(url)
    length = response.headers.get("Content-Length")

    def chunks() -> Iterator[bytes]:
        with response:
            while chunk := response.read(CHUNK_BYTES):
                yield chunk

    return Transfer(int(length) if length else None, chunks())


def cache_directory(environ: Mapping[str, str]) -> Path:
    """Where downloads are kept: the configured directory, else the XDG cache."""
    if configured := environ.get(CACHE_ENV):
        return Path(configured)
    base = environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "stormfork"


def megabytes(size_bytes: int) -> str:
    """Format a byte count for a person reading the terminal."""
    return f"{size_bytes / BYTES_PER_MB:.1f} MB"


def offline_error(source: Source) -> UserError:
    """The failure for an input that is not cached while downloads are off."""
    msg = f"no cached {source.title}, and {OFFLINE_ENV} is set"
    return UserError(msg, f"run stormfork fetch, or pass a local file to --{source.flag}", 2)


def open_transfer(source: Source, transport: Transport) -> Transfer:
    """Start a source's download, naming what the publisher's server answered.

    Raises:
        UserError: The file has moved, or the server cannot be reached.
    """
    host = urlsplit(source.url).hostname
    try:
        return transport(source.url)
    except HTTPError as error:
        if error.code in {HTTPStatus.NOT_FOUND, HTTPStatus.GONE}:
            msg = f"{source.url} answered HTTP {error.code}: {source.title} moved or removed"
            raise UserError(msg, f"pass a copy you trust to --{source.flag}", 3) from error
        msg = f"could not reach {host} for the {source.title} (HTTP {error.code})"
        next_step = f"check the connection and retry, or pass a local file to --{source.flag}"
        raise UserError(msg, next_step, 3) from error
    except (URLError, TimeoutError) as error:
        msg = f"could not reach {host} for the {source.title} ({error})"
        next_step = f"check the connection and retry, or pass a local file to --{source.flag}"
        raise UserError(msg, next_step, 3) from error


def check_free_space(cache: Path, size_bytes: int) -> None:
    """Refuse a download that would leave less than the headroom free.

    Raises:
        UserError: The cache's disk has too little free space.
    """
    free = shutil.disk_usage(cache).free
    if free < size_bytes + DISK_HEADROOM_BYTES:
        msg = (
            f"{cache} has {megabytes(free)} free; the download needs {megabytes(size_bytes)} "
            f"plus {megabytes(DISK_HEADROOM_BYTES)} headroom"
        )
        next_step = f"free space there, or set {CACHE_ENV} to a directory on a larger disk"
        raise UserError(msg, next_step, 1)


def stopped_error(source: Source, cache: Path, written: int, error: Exception) -> UserError:
    """Describe a transfer that ended early, telling a full disk from a lost connection."""
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        msg = f"the disk filled while writing {source.filename} to {cache}"
        next_step = f"free space there, or set {CACHE_ENV} to a directory on a larger disk"
        return UserError(msg, next_step, 1)
    msg = f"download of {source.filename} stopped after {megabytes(written)} ({error})"
    return UserError(msg, "rerun the same command; the partial file was removed", 3)


def verify(source: Source, transfer: Transfer, written: int, sha256: str) -> None:
    """Check a finished transfer's length and, for a pinned source, its digest.

    Raises:
        UserError: The body is shorter or longer than announced, or its digest differs.
    """
    if transfer.size_bytes is not None and written != transfer.size_bytes:
        msg = f"{source.filename} ended after {written} of {transfer.size_bytes} bytes"
        raise UserError(msg, "rerun the same command; the partial file was removed", 3)
    if source.sha256 is not None and sha256 != source.sha256:
        host = urlsplit(source.url).hostname
        msg = f"{source.filename} from {host} has SHA-256 {sha256}, pinned {source.sha256}"
        next_step = (
            f"rerun stormfork fetch; if it fails again, pass a copy you trust to --{source.flag}"
        )
        raise UserError(msg, next_step, 3)


def save(source: Source, transfer: Transfer, cache: Path) -> Path:
    """Stream a transfer into the cache, keeping the file only once it verifies.

    Raises:
        UserError: The transfer stopped, the disk filled, or the body failed verification.
    """
    destination = cache / source.filename
    digest = hashlib.sha256()
    written = 0
    part = NamedTemporaryFile(  # noqa: SIM115 - closed below, before the rename
        dir=cache, prefix=f".{source.filename}.", suffix=".part", delete=False
    )
    partial = Path(part.name)
    try:
        with part:
            try:
                for chunk in transfer.chunks:
                    part.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            except (OSError, HTTPException) as error:
                raise stopped_error(source, cache, written, error) from error
        verify(source, transfer, written, digest.hexdigest())
        partial.chmod(CACHE_FILE_MODE)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def is_offline(environ: Mapping[str, str]) -> bool:
    """Downloads are off when the offline variable is set to anything but empty or 0."""
    return environ.get(OFFLINE_ENV, "") not in {"", "0"}


def resolve(
    source: Source,
    *,
    transport: Transport = https_transfer,
    environ: Mapping[str, str] = os.environ,
) -> Path:
    """Return the cached copy of a source, downloading and verifying it first if absent.

    Raises:
        UserError: Downloads are off and the file is not cached, or the download failed.
    """
    cache = cache_directory(environ)
    path = cache / source.filename
    if path.is_file():
        return path
    if is_offline(environ):
        raise offline_error(source)
    cache.mkdir(parents=True, exist_ok=True)
    transfer = open_transfer(source, transport)
    size = megabytes(transfer.size_bytes or 0)
    sys.stderr.write(f"downloading {source.title}, {size}, from {source.url}\n")
    check_free_space(cache, transfer.size_bytes or 0)
    return save(source, transfer, cache)


def fetch_all(
    *, transport: Transport = https_transfer, environ: Mapping[str, str] = os.environ
) -> str:
    """Download every input that is not cached, listing sizes before the first byte.

    Raises:
        UserError: Downloads are off while an input is missing, or a download failed.
    """
    cache = cache_directory(environ)
    missing = [source for source in SOURCES if not (cache / source.filename).is_file()]
    if missing and is_offline(environ):
        raise offline_error(missing[0])
    cache.mkdir(parents=True, exist_ok=True)
    transfers = {source.flag: open_transfer(source, transport) for source in missing}
    for source in SOURCES:
        transfer = transfers.get(source.flag)
        if transfer is None:
            sys.stderr.write(f"{source.title}: cached\n")
        else:
            host = urlsplit(source.url).hostname
            sys.stderr.write(f"{source.title}: {megabytes(transfer.size_bytes or 0)} from {host}\n")
    total = sum(transfer.size_bytes or 0 for transfer in transfers.values())
    if transfers:
        sys.stderr.write(f"downloading {megabytes(total)}\n")
        check_free_space(cache, total)
    for source in missing:
        save(source, transfers[source.flag], cache)
    lines = [f"{source.filename}: {source.credit}" for source in SOURCES]
    return "\n".join([f"cache: {cache}", *lines])
