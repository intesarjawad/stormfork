import hashlib
import json
import threading
import zipfile
from collections.abc import Iterator
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts import fetch_atlases
from stormfork import cli_remote
from stormfork.downloads import OFFLINE_ENV
from stormfork.export.manifest import verify_manifest
from tests.test_cli import invoke
from tests.test_hindcast_export import write_ibtracs, write_run_directory
from tests.test_manifest import example_export
from tests.test_verify_remote import QuietHandler

RELEASE = "atlases-20261110"
EXPERIMENT = "synthetic_comparison"


def zip_export(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(source))


@pytest.fixture
def checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A clone after an export and a release upload: the zip in dist/atlases and the hero."""
    source = tmp_path / "export"
    source.mkdir()
    example_export(source)
    clone = tmp_path / "clone"
    (clone / "dist/atlases").mkdir(parents=True, exist_ok=True)
    zip_export(source, clone / f"dist/atlases/{EXPERIMENT}.zip")
    (clone / "site").mkdir()
    (clone / "site/hero.json").write_text(json.dumps({"id": EXPERIMENT, "landfall_step": 2}))
    monkeypatch.chdir(clone)
    return clone


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Serve the clone's dist/atlases as the release's assets from a loopback server."""
    assets = tmp_path / "clone/dist/atlases"
    assets.mkdir(parents=True, exist_ok=True)
    served = tmp_path / "served"
    served.mkdir()
    (served / RELEASE).symlink_to(assets)
    http = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=served))
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    monkeypatch.setattr(cli_remote, "RELEASE_URL", f"http://127.0.0.1:{http.server_port}")
    monkeypatch.delenv(OFFLINE_ENV)
    try:
        yield
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.fixture
def release(checkout: Path, server: None) -> Path:
    del server
    return checkout / f"dist/atlases/{EXPERIMENT}.zip"


def test_lock_pins_each_exported_zip_under_the_release(
    checkout: Path, capsys: pytest.CaptureFixture[str]
):
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    assert capsys.readouterr().out == f"site/atlases.lock.json: 1 experiments pinned in {RELEASE}\n"
    archive = checkout / f"dist/atlases/{EXPERIMENT}.zip"
    assert json.loads((checkout / "site/atlases.lock.json").read_text()) == {
        "hero": {"id": EXPERIMENT, "landfall_step": 2},
        "release": RELEASE,
        "experiments": {
            EXPERIMENT: {
                "zip": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        },
    }


def test_fetch_unpacks_every_pinned_experiment_beside_the_tracked_files(
    release: Path, checkout: Path, capsys: pytest.CaptureFixture[str]
):
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    published = checkout / "site/public/experiments" / EXPERIMENT
    published.mkdir(parents=True)
    (published / "reproductions.json").write_text("[]\n")
    capsys.readouterr()
    assert fetch_atlases.main([]) == 0
    assert capsys.readouterr().out == "1 experiments unpacked into site/public/experiments\n"
    assert verify_manifest(published).id == EXPERIMENT
    assert (published / "reproductions.json").read_text() == "[]\n"
    assert release.is_file()


def test_fetch_fails_when_one_byte_of_a_zip_changes(
    release: Path, checkout: Path, capsys: pytest.CaptureFixture[str]
):
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    data = bytearray(release.read_bytes())
    data[len(data) // 2] ^= 1
    release.write_bytes(data)
    capsys.readouterr()
    assert fetch_atlases.main([]) == 1
    assert "does not match the lock" in capsys.readouterr().err
    assert not (checkout / "site/public/experiments").exists()


def test_fetch_fails_when_the_pinned_release_asset_is_absent(
    release: Path, checkout: Path, capsys: pytest.CaptureFixture[str]
):
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    release.unlink()
    assert fetch_atlases.main([]) != 0
    assert "could not reach" in capsys.readouterr().err
    assert not (checkout / "site/public/experiments").exists()


@pytest.mark.usefixtures("checkout")
def test_fetch_reads_nothing_while_downloads_are_off(capsys: pytest.CaptureFixture[str]):
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    capsys.readouterr()
    assert fetch_atlases.main([]) == 2
    error = capsys.readouterr().err
    assert error.startswith(f"error: {OFFLINE_ENV} is set, so nothing was fetched from ")
    assert error.endswith(f"next: unset {OFFLINE_ENV} and retry\n")


@pytest.mark.usefixtures("checkout")
def test_fetch_without_a_lock_leaves_the_site_on_its_fixtures(
    capsys: pytest.CaptureFixture[str],
):
    assert fetch_atlases.main([]) == 0
    assert capsys.readouterr() == ("no atlases lock; fixtures only\n", "")


def test_lock_needs_an_exported_zip(checkout: Path, capsys: pytest.CaptureFixture[str]):
    (checkout / f"dist/atlases/{EXPERIMENT}.zip").unlink()
    assert fetch_atlases.main(["--lock", RELEASE]) == 1
    assert capsys.readouterr().err == (
        "error: dist/atlases holds no zip; export the experiments first\n"
    )


@pytest.mark.usefixtures("server")
def test_an_export_round_trips_through_the_lock_the_release_and_verify(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    hindcast = "hindcast_synthetic-2024100700"
    write_run_directory(tmp_path / "run")
    # The model output beside the tracks stays out of the release.
    (tmp_path / "run/baseline/forecast.nc").write_bytes(b"forecast")
    ibtracs = write_ibtracs(tmp_path)
    clone = tmp_path / "clone"
    (clone / "site").mkdir(parents=True)
    (clone / "site/hero.json").write_text(json.dumps({"id": hindcast}))
    monkeypatch.chdir(clone)
    archive = clone / f"dist/atlases/{hindcast}.zip"
    exported: list[bytes] = []
    for out in ("../first", "../second"):
        invoke(monkeypatch, "export", "../run", "--out", out, "--ibtracs", str(ibtracs))
        exported.append(archive.read_bytes())
    assert exported[0] == exported[1]
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
    names = [entry.filename for entry in entries]
    assert names == sorted(names)
    assert "manifest.json" in names
    assert "baseline/forecast.nc" not in names
    assert {entry.date_time for entry in entries} == {(1980, 1, 1, 0, 0, 0)}
    assert fetch_atlases.main(["--lock", RELEASE]) == 0
    assert fetch_atlases.main([]) == 0
    capsys.readouterr()
    invoke(monkeypatch, "verify", hindcast)
    lines = capsys.readouterr().out.splitlines()
    assert (lines[0], lines[-1]) == (hindcast, "PASS")
