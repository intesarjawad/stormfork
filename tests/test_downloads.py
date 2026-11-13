"""The download cache, with a fake transport in place of the network."""

import errno
import hashlib
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError

import pytest

from stormfork import downloads
from stormfork.downloads import Source, Transfer, Transport
from stormfork.errors import UserError
from tests.test_cli import invoke

BODY = b"0123456789" * 1000
SOURCE = Source(
    flag="weights",
    title="model weights",
    url="https://example.org/files/weights.npz",
    filename="weights.npz",
    sha256=hashlib.sha256(BODY).hexdigest(),
    credit="Weights CC BY 4.0.",
)
UNPINNED = Source(
    flag="ibtracs",
    title="IBTrACS",
    url="https://example.org/files/ibtracs.nc",
    filename="ibtracs.nc",
    sha256=None,
    credit="Public domain.",
)


class FakeTransport(NamedTuple):
    transport: Transport
    requested: list[str]


def serve(
    body: bytes = BODY, *, size: int | None = len(BODY), fail_after: BaseException | None = None
) -> FakeTransport:
    """Serve one body in two chunks, optionally failing after the first."""
    requested: list[str] = []

    def chunks() -> Iterator[bytes]:
        half = len(body) // 2
        yield body[:half]
        if fail_after is not None:
            raise fail_after
        yield body[half:]

    def transport(url: str) -> Transfer:
        requested.append(url)
        return Transfer(size, chunks())

    return FakeTransport(transport, requested)


def refuse(error: Exception) -> Transport:
    def transport(url: str) -> Transfer:
        del url
        raise error

    return transport


@pytest.fixture
def environ(tmp_path: Path) -> dict[str, str]:
    return {downloads.CACHE_ENV: str(tmp_path / "cache")}


def cache_files(environ: dict[str, str]) -> list[str]:
    cache = Path(environ[downloads.CACHE_ENV])
    return sorted(path.name for path in cache.iterdir()) if cache.exists() else []


def test_a_cached_file_is_used_without_a_request(environ: dict[str, str]):
    cache = Path(environ[downloads.CACHE_ENV])
    cache.mkdir()
    (cache / "weights.npz").write_bytes(b"already here")
    fake = serve()
    assert downloads.resolve(SOURCE, transport=fake.transport, environ=environ) == (
        cache / "weights.npz"
    )
    assert fake.requested == []


def test_a_missing_file_is_downloaded_and_verified(
    environ: dict[str, str], capsys: pytest.CaptureFixture[str]
):
    fake = serve()
    path = downloads.resolve(SOURCE, transport=fake.transport, environ=environ)
    assert fake.requested == [SOURCE.url]
    assert path.read_bytes() == BODY
    assert cache_files(environ) == ["weights.npz"]
    assert capsys.readouterr().err == (f"downloading model weights, 0.0 MB, from {SOURCE.url}\n")


def test_a_checksum_mismatch_keeps_nothing(environ: dict[str, str]):
    fake = serve(BODY[::-1])
    with pytest.raises(UserError, match=r"weights.npz from example.org has SHA-256") as raised:
        downloads.resolve(SOURCE, transport=fake.transport, environ=environ)
    assert raised.value.exit_code == 3
    assert raised.value.next_step == (
        "rerun stormfork fetch; if it fails again, pass a copy you trust to --weights"
    )
    assert cache_files(environ) == []


def test_a_transfer_shorter_than_announced_keeps_nothing(environ: dict[str, str]):
    fake = serve(size=len(BODY) + 1)
    with pytest.raises(UserError, match=r"ibtracs.nc ended after 10000 of 10001 bytes"):
        downloads.resolve(UNPINNED, transport=fake.transport, environ=environ)
    assert cache_files(environ) == []


def test_an_unpinned_file_is_kept_when_complete(environ: dict[str, str]):
    path = downloads.resolve(UNPINNED, transport=serve().transport, environ=environ)
    assert path.read_bytes() == BODY


def test_a_dropped_connection_keeps_no_partial_file(environ: dict[str, str]):
    fake = serve(fail_after=ConnectionResetError("reset by peer"))
    with pytest.raises(UserError, match=r"download of weights.npz stopped") as raised:
        downloads.resolve(SOURCE, transport=fake.transport, environ=environ)
    assert raised.value.exit_code == 3
    assert raised.value.next_step == "rerun the same command; the partial file was removed"
    assert cache_files(environ) == []


def test_an_interrupted_download_keeps_no_partial_file(environ: dict[str, str]):
    fake = serve(fail_after=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        downloads.resolve(SOURCE, transport=fake.transport, environ=environ)
    assert cache_files(environ) == []


def test_a_disk_that_fills_during_the_write_is_named(environ: dict[str, str]):
    fake = serve(fail_after=OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(UserError, match=r"the disk filled while writing weights.npz") as raised:
        downloads.resolve(SOURCE, transport=fake.transport, environ=environ)
    assert raised.value.next_step == (
        f"free space there, or set {downloads.CACHE_ENV} to a directory on a larger disk"
    )
    assert cache_files(environ) == []


def test_too_little_free_space_stops_before_writing(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str]
):
    class Usage(NamedTuple):
        total: int
        used: int
        free: int

    def disk_usage(path: Path) -> Usage:
        del path
        return Usage(0, 0, len(BODY))

    monkeypatch.setattr(downloads.shutil, "disk_usage", disk_usage)
    with pytest.raises(UserError, match=r"has 0.0 MB free") as raised:
        downloads.resolve(SOURCE, transport=serve().transport, environ=environ)
    assert raised.value.exit_code == 1
    assert cache_files(environ) == []


def test_offline_mode_fails_before_any_request(environ: dict[str, str]):
    fake = serve()
    offline = environ | {downloads.OFFLINE_ENV: "1"}
    with pytest.raises(UserError, match="no cached model weights") as raised:
        downloads.resolve(SOURCE, transport=fake.transport, environ=offline)
    assert raised.value.next_step == "run stormfork fetch, or pass a local file to --weights"
    assert fake.requested == []


def test_offline_mode_still_reads_the_cache(environ: dict[str, str]):
    cache = Path(environ[downloads.CACHE_ENV])
    cache.mkdir()
    (cache / "weights.npz").write_bytes(BODY)
    offline = environ | {downloads.OFFLINE_ENV: "1"}
    assert downloads.resolve(SOURCE, transport=refuse(URLError("no")), environ=offline).is_file()


@pytest.mark.parametrize("value", ["", "0"])
def test_an_empty_or_zero_offline_variable_allows_downloads(environ: dict[str, str], value: str):
    online = environ | {downloads.OFFLINE_ENV: value}
    assert downloads.resolve(SOURCE, transport=serve().transport, environ=online).is_file()


def test_no_network_names_the_host_and_the_local_override(environ: dict[str, str]):
    transport = refuse(URLError("nodename nor servname provided"))
    with pytest.raises(UserError, match=r"could not reach example.org") as raised:
        downloads.resolve(SOURCE, transport=transport, environ=environ)
    assert raised.value.exit_code == 3
    assert raised.value.next_step == (
        "check the connection and retry, or pass a local file to --weights"
    )


@pytest.mark.parametrize("status", [404, 410])
def test_a_moved_source_says_so(environ: dict[str, str], status: int):
    transport = refuse(HTTPError(SOURCE.url, status, "Not Found", Message(), None))
    with pytest.raises(UserError, match=f"answered HTTP {status}") as raised:
        downloads.resolve(SOURCE, transport=transport, environ=environ)
    assert "moved or removed" in raised.value.what
    assert raised.value.next_step == "pass a copy you trust to --weights"


def test_fetch_lists_sizes_before_downloading_only_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(downloads, "SOURCES", (SOURCE, UNPINNED))
    cache = Path(environ[downloads.CACHE_ENV])
    cache.mkdir()
    (cache / "ibtracs.nc").write_bytes(BODY)
    fake = serve(size=2_500_000 + len(BODY))
    with pytest.raises(UserError, match="ended after"):
        downloads.fetch_all(transport=fake.transport, environ=environ)
    assert fake.requested == [SOURCE.url]
    assert capsys.readouterr().err.splitlines() == [
        "model weights: 2.5 MB from example.org",
        "IBTrACS: cached",
        "downloading 2.5 MB",
    ]
    output = downloads.fetch_all(transport=serve().transport, environ=environ)
    assert output.splitlines() == [
        f"cache: {cache}",
        "weights.npz: Weights CC BY 4.0.",
        "ibtracs.nc: Public domain.",
    ]
    assert (cache / "weights.npz").read_bytes() == BODY


def test_fetch_refuses_to_download_offline(environ: dict[str, str]):
    offline = environ | {downloads.OFFLINE_ENV: "1"}
    with pytest.raises(UserError, match="no cached model weights"):
        downloads.fetch_all(transport=serve().transport, environ=offline)


def test_the_default_cache_follows_xdg(tmp_path: Path):
    environ = {"XDG_CACHE_HOME": str(tmp_path)}
    assert downloads.cache_directory(environ) == tmp_path / "stormfork"
    assert downloads.cache_directory({downloads.CACHE_ENV: "here"}) == Path("here")


def test_every_pin_is_https_and_fixed_files_carry_a_digest():
    for source in downloads.SOURCES:
        assert source.url.startswith("https://")
    assert [source.flag for source in downloads.SOURCES if source.sha256] == ["weights", "sample"]


def test_the_fetch_command_reports_the_cache_and_each_credit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], offline_cache: Path
):
    for source in downloads.SOURCES:
        (offline_cache / source.filename).write_bytes(b"cached")
    invoke(monkeypatch, "fetch")
    # The command wraps at 80 columns, so compare without whitespace.
    printed = "".join(capsys.readouterr().out.split())
    expected = [
        f"cache: {offline_cache}",
        *(f"{s.filename}: {s.credit}" for s in downloads.SOURCES),
    ]
    assert printed == "".join("".join(expected).split())


def test_the_fetch_command_offline_names_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "fetch")
    assert raised.value.code == 2
    assert capsys.readouterr().err.splitlines()[-2:] == [
        f"error: no cached model weights, and {downloads.OFFLINE_ENV} is set",
        "next: run stormfork fetch, or pass a local file to --weights",
    ]
