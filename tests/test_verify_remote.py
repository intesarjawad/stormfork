import hashlib
import json
import threading
import zipfile
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stormfork import cli_remote
from stormfork.downloads import OFFLINE_ENV
from tests.test_cli import invoke
from tests.test_cli_output import GOLDEN
from tests.test_manifest import example_export

RELEASE = "atlases-test"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Path, str]]:
    experiment = tmp_path / "source"
    experiment.mkdir()
    example_export(experiment)
    served = tmp_path / "served"
    (served / RELEASE).mkdir(parents=True)
    archive = served / RELEASE / "synthetic_comparison.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(experiment.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(experiment))
    lock = {
        "hero": {"id": "synthetic_comparison"},
        "release": RELEASE,
        "experiments": {
            "synthetic_comparison": {
                "zip": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        },
    }
    (served / "atlases.lock.json").write_text(json.dumps(lock))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=served))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(cli_remote, "LOCK_URL", f"{base}/atlases.lock.json")
    monkeypatch.setattr(cli_remote, "RELEASE_URL", base)
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.delenv(OFFLINE_ENV)
    try:
        yield archive, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("arguments", [(), ("synthetic_comparison",)])
def test_remote_experiment_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    release: tuple[Path, str],
    arguments: tuple[str, ...],
):
    _, base = release
    invoke(monkeypatch, "verify", *arguments)
    captured = capsys.readouterr()
    assert captured.out == GOLDEN
    assert captured.err.splitlines() == [
        f"fetch {base}/atlases.lock.json",
        f"fetch {base}/{RELEASE}/synthetic_comparison.zip",
    ]


def test_one_changed_byte_in_the_archive_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], release: tuple[Path, str]
):
    archive, _ = release
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 1
    archive.write_bytes(data)
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "synthetic_comparison")
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: synthetic_comparison.zip SHA-2" in captured.err
    with pytest.raises(ValueError, match="does not match the lock"):
        invoke(monkeypatch, "--debug", "verify", "synthetic_comparison")


@pytest.mark.usefixtures("release")
def test_an_id_missing_from_the_lock_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "absent")
    assert raised.value.code == 2
    assert 'error: experiment "absent" is not in the release lock\n' in capsys.readouterr().err


def test_remote_http_is_limited_to_loopback():
    with pytest.raises(ValueError, match="HTTPS"):
        cli_remote.fetch_bytes("http://example.com/atlases.lock.json")


def test_an_archive_that_escapes_its_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], release: tuple[Path, str]
):
    archive, _ = release
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("../evil", b"x")
    lock = archive.parent.parent / "atlases.lock.json"
    document = json.loads(lock.read_text())
    document["experiments"]["synthetic_comparison"]["sha256"] = hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    lock.write_text(json.dumps(document))
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "synthetic_comparison")
    assert raised.value.code == 1
    assert "not a plain relative path" in capsys.readouterr().err
    assert not (archive.parent.parent.parent / "evil").exists()


def test_the_lock_in_the_checkout_is_read_before_the_remote_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], release: tuple[Path, str]
):
    archive, base = release
    served_lock = archive.parent.parent / "atlases.lock.json"
    Path("site").mkdir()
    served_lock.rename("site/atlases.lock.json")
    invoke(monkeypatch, "verify", "synthetic_comparison")
    captured = capsys.readouterr()
    assert captured.out == GOLDEN
    assert captured.err.splitlines() == [
        "lock site/atlases.lock.json",
        f"fetch {base}/{RELEASE}/synthetic_comparison.zip",
    ]


def test_a_missing_remote_lock_means_nothing_is_published_yet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], release: tuple[Path, str]
):
    archive, _ = release
    (archive.parent.parent / "atlases.lock.json").unlink()
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "synthetic_comparison")
    assert raised.value.code == 2
    assert capsys.readouterr().err.splitlines()[-2:] == [
        "error: no published experiments yet",
        "next: stormfork scenarios list",
    ]


@pytest.mark.usefixtures("release")
def test_nothing_is_fetched_while_downloads_are_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv(OFFLINE_ENV, "1")
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "verify", "synthetic_comparison")
    assert raised.value.code == 2
    error, next_step = capsys.readouterr().err.splitlines()
    assert error.startswith(f"error: {OFFLINE_ENV} is set")
    assert next_step == f"next: unset {OFFLINE_ENV} and retry"
