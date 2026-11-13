import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.publish_experiment import check_experiment, main, update_lock
from stormfork.cli_remote import Lock
from tests.test_reproduction import export_tracks, published_tracks

EXPERIMENT = "milton_noise_1pct"


def entry(by: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "by": by,
        "runtime": "cpu: Darwin arm64",
        "date": "2026-11-06",
        "code_version": "fixture-v1",
        "max_track_diff_km": 0.5,
        "passed": passed,
    }


def zip_directory(directory: Path, archive: Path, replace: dict[str, bytes] | None = None) -> Path:
    replace = replace or {}
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                name = path.relative_to(directory).as_posix()
                bundle.writestr(name, replace.get(name, path.read_bytes()))
        for name, data in replace.items():
            if name not in bundle.namelist():
                bundle.writestr(name, data)
    return archive


class Fixture:
    def __init__(self, root: Path, *, run_by: str = "alice") -> None:
        self.exported = export_tracks(
            root / "exported", published_tracks(), {"id": EXPERIMENT, "run_by": run_by}
        )
        self.experiments = root / "experiments"
        self.directory = self.experiments / EXPERIMENT
        self.directory.mkdir(parents=True)
        (self.directory / "manifest.json").write_bytes(
            (self.exported / "manifest.json").read_bytes()
        )
        self.ledger([entry("bob")])
        self.archive = zip_directory(self.exported, root / f"{EXPERIMENT}.zip")
        self.workdir = root / "unpacked"
        self.runs = 0

    def ledger(self, entries: list[dict[str, object]]) -> None:
        (self.directory / "reproductions.json").write_text(json.dumps(entries))

    def check(self, archive: Path | None = None) -> None:
        self.runs += 1
        workdir = self.workdir / str(self.runs)
        check_experiment(archive or self.archive, EXPERIMENT, self.experiments, workdir)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_a_matching_reproduced_archive_passes(fixture: Fixture):
    fixture.check()
    assert (fixture.workdir / "1" / "manifest.json").is_file()


def test_a_changed_file_is_refused_on_its_hash(tmp_path: Path, fixture: Fixture):
    other = tmp_path / "other.png"
    Image.new("L", (1805, 1629), 101).save(other)
    archive = zip_directory(
        fixture.exported, tmp_path / "changed.zip", {"fork/mslp.png": other.read_bytes()}
    )
    with pytest.raises(ValueError, match=r"fork/mslp\.png SHA-256 .* does not match the manifest"):
        fixture.check(archive)


def test_a_self_reproduction_is_refused(fixture: Fixture):
    fixture.ledger([entry("alice")])
    with pytest.raises(ValueError, match="no passing reproduction by anyone but alice"):
        fixture.check()
    fixture.ledger([entry("Alice")])
    with pytest.raises(ValueError, match="no passing reproduction by anyone but alice"):
        fixture.check()
    assert not fixture.workdir.exists()


def test_only_a_passing_reproduction_by_someone_else_counts(fixture: Fixture):
    fixture.ledger([entry("alice"), entry("bob", passed=False)])
    with pytest.raises(ValueError, match="no passing reproduction"):
        fixture.check()
    fixture.ledger([entry("alice"), entry("bob", passed=False), entry("carol")])
    fixture.check()


def test_missing_ledger_runner_or_merge_is_refused(tmp_path: Path, fixture: Fixture):
    (fixture.directory / "reproductions.json").unlink()
    with pytest.raises(ValueError, match=r"no reproductions\.json"):
        fixture.check()
    anonymous = Fixture(tmp_path / "anonymous")
    (anonymous.exported / "manifest.json").unlink()
    export_tracks(tmp_path / "plain", published_tracks(), {"id": EXPERIMENT})
    (anonymous.directory / "manifest.json").write_bytes(
        (tmp_path / "plain" / "manifest.json").read_bytes()
    )
    with pytest.raises(ValueError, match="no run_by"):
        anonymous.check()
    (fixture.directory / "manifest.json").unlink()
    with pytest.raises(ValueError, match="is not merged yet"):
        fixture.check()


def test_an_archive_with_another_manifest_or_extra_files_is_refused(
    tmp_path: Path, fixture: Fixture
):
    changed = (fixture.exported / "manifest.json").read_text().replace("Synthetic", "Changed")
    archive = zip_directory(
        fixture.exported, tmp_path / "a.zip", {"manifest.json": changed.encode()}
    )
    with pytest.raises(ValueError, match="differs from the merged one"):
        fixture.check(archive)
    archive = zip_directory(fixture.exported, tmp_path / "b.zip", {"extra.txt": b"x"})
    with pytest.raises(ValueError, match=r"does not name: \['extra.txt'\]"):
        fixture.check(archive)


def test_an_unsafe_archive_is_refused_before_anything_is_checked(tmp_path: Path, fixture: Fixture):
    archive = zip_directory(fixture.exported, tmp_path / "evil.zip", {"../evil": b"x"})
    with pytest.raises(ValueError, match="plain relative path"):
        fixture.check(archive)
    assert not (tmp_path / "evil").exists()


def test_the_command_line_reports_a_refusal(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, fixture: Fixture
):
    arguments = ["check", EXPERIMENT, str(fixture.archive), "--workdir", str(fixture.workdir)]
    arguments += ["--experiments", str(fixture.experiments)]
    assert main(arguments) == 0
    assert "matches the merged manifest" in capsys.readouterr().out
    fixture.ledger([entry("alice")])
    assert main([*arguments[:4], str(tmp_path / "again"), *arguments[5:]]) == 1
    assert "no passing reproduction by anyone but alice" in capsys.readouterr().err


def test_the_lock_pins_the_archive_under_the_new_release(tmp_path: Path, fixture: Fixture):
    lock = tmp_path / "atlases.lock.json"
    previous = {"zip": "other.zip", "sha256": "a" * 64}
    lock.write_text(
        json.dumps(
            {
                "hero": {"id": "other"},
                "release": "atlases-20261101",
                "experiments": {"other": previous},
            }
        )
    )
    assert (
        main(
            [
                "lock",
                EXPERIMENT,
                str(fixture.archive),
                "--release",
                "atlases-20261107",
                "--lock",
                str(lock),
            ]
        )
        == 0
    )
    result = Lock.model_validate_json(lock.read_bytes())
    assert result.release == "atlases-20261107"
    assert result.experiments["other"].model_dump() == previous
    assert result.experiments[EXPERIMENT].zip == f"{EXPERIMENT}.zip"
    assert (
        result.experiments[EXPERIMENT].sha256
        == hashlib.sha256(fixture.archive.read_bytes()).hexdigest()
    )
    assert result.hero.id == "other"
    update_lock(lock, EXPERIMENT, fixture.archive, "atlases-20261108")
    assert Lock.model_validate_json(lock.read_bytes()).release == "atlases-20261108"
