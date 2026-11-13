import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from pydantic import ValidationError

from stormfork.cli import main
from stormfork.cli_data import parse_time
from stormfork.cli_verify import has_observed
from stormfork.export.divergence import RunName, compute_divergence
from stormfork.export.manifest import (
    HINDCAST_LIMIT,
    ExperimentMetadata,
    export_experiment,
    verify_manifest,
)
from stormfork.export.observed import observed_track
from stormfork.export.quantise import FILE_NAMES
from tests.test_divergence import synthetic_tracks
from tests.test_manifest import example_export, example_metadata

if TYPE_CHECKING:
    import pandas as pd

INIT = parse_time("2024-10-07T00:00:00")
IBTRACS = """SID,ISO_TIME,USA_LAT,USA_LON,USA_WIND,USA_PRES
 , ,degrees_north,degrees_east,kts,mb
S1,2024-10-06 18:00:00,19.0,-59.0,90,960
S1,2024-10-07 00:00:00,20.0,-60.0,100,950
S1,2024-10-07 03:00:00,20.5,-60.5,100,950
S1,2024-10-07 06:00:00,21.0,-61.0,110,945
S1,2024-10-07 12:00:00,22.0,-62.0, ,970
S1,2024-10-07 18:00:00,23.0,-63.0,90,960
S1,2024-10-08 00:00:00,24.0,-64.0,80,-9999
S1,2024-10-08 06:00:00,25.0,-65.0,70,980
S1,2024-10-08 12:00:00,26.0,-66.0,60,985
S2,2024-10-07 00:00:00,10.0,-10.0,50,990
"""
KNOT = 1852 / 3600


def hindcast_metadata(limits: list[str] | None = None) -> ExperimentMetadata:
    document = example_metadata().model_dump(mode="json")
    document.update(
        kind="hindcast",
        id="hindcast_synthetic-2024100700",
        parent_id="hindcast_synthetic",
        limits=[HINDCAST_LIMIT] if limits is None else limits,
        perturbation={"type": "none", "params": {}},
        diagnostics={},
    )
    document["init"].update(source="era5", time=INIT.isoformat())
    document["init"]["storm"]["ibtracs_sid"] = "S1"
    return ExperimentMetadata.model_validate(document)


def write_ibtracs(directory: Path) -> Path:
    path = directory / "ibtracs.csv"
    path.write_text(IBTRACS)
    return path


def write_run_directory(directory: Path) -> None:
    (directory / "baseline").mkdir(parents=True)
    synthetic_tracks().to_csv(directory / "baseline" / "tracks.csv", index=False)
    for file in FILE_NAMES:
        Image.new("L", (1805, 1629), 100).save(directory / "baseline" / f"{file}.png")
    (directory / "metadata.json").write_text(hindcast_metadata().model_dump_json())


def test_observed_track_keeps_six_hourly_complete_samples_from_init(tmp_path: Path):
    frame = observed_track(write_ibtracs(tmp_path), "S1", INIT, 5)
    assert frame.step.tolist() == [0, 1, 3, 5]
    assert frame.member.tolist() == [0, 0, 0, 0]
    assert frame.track_id.tolist() == ["S1"] * 4
    assert frame.lat.tolist() == [20.0, 21.0, 23.0, 25.0]
    assert frame.lon.tolist() == [-60.0, -61.0, -63.0, -65.0]
    assert frame.mslp_pa.tolist() == [95000.0, 94500.0, 96000.0, 98000.0]
    assert frame.wind_ms.tolist() == pytest.approx([100 * KNOT, 110 * KNOT, 90 * KNOT, 70 * KNOT])
    assert frame.valid_time.dt.strftime("%Y-%m-%d %H").tolist() == [
        "2024-10-07 00",
        "2024-10-07 06",
        "2024-10-07 18",
        "2024-10-08 06",
    ]


def test_observed_track_accepts_a_timezone_aware_init(tmp_path: Path):
    aware = datetime.fromisoformat("2024-10-07T00:00:00+00:00")
    assert observed_track(write_ibtracs(tmp_path), "S1", aware, 1).step.tolist() == [0, 1]


def test_observed_track_rejects_a_missing_storm_and_missing_columns(tmp_path: Path):
    path = write_ibtracs(tmp_path)
    with pytest.raises(ValueError, match="no usable samples for S9"):
        observed_track(path, "S9", INIT, 4)
    path.write_text("SID,ISO_TIME\nS1,2024-10-07 00:00:00\n")
    with pytest.raises(ValueError, match="missing columns: USA_LAT, USA_LON, USA_PRES, USA_WIND"):
        observed_track(path, "S1", INIT, 4)


def test_hindcast_export_publishes_observed_truth_and_the_reconstruction_limit(tmp_path: Path):
    tracks: dict[RunName, pd.DataFrame] = {"baseline": synthetic_tracks()}
    for file in FILE_NAMES:
        (tmp_path / "baseline").mkdir(exist_ok=True)
        Image.new("L", (1805, 1629), 100).save(tmp_path / "baseline" / f"{file}.png")
    observed = observed_track(write_ibtracs(tmp_path), "S1", INIT, 4)
    manifest = export_experiment(
        tmp_path,
        hindcast_metadata(),
        tracks,
        compute_divergence(tracks),
        observed=observed,
    )
    assert verify_manifest(tmp_path) == manifest
    assert HINDCAST_LIMIT in manifest.limits
    assert set(manifest.runs) == {"baseline"}
    published = json.loads((tmp_path / "tracks.json").read_text())
    truth = [sample for sample in published if sample["run"] == "observed"]
    assert [sample["step"] for sample in truth] == [0, 1, 3]
    assert {sample["track_id"] for sample in truth} == {"S1"}
    assert sum(sample["run"] == "baseline" for sample in published) == 40
    assert "observed" in (tmp_path / "tracks.csv").read_text()
    assert has_observed(tmp_path, manifest)


def test_a_hindcast_without_the_reconstruction_limit_is_rejected():
    with pytest.raises(ValidationError, match="limits must include: Initial conditions"):
        hindcast_metadata(["Synthetic tracks do not demonstrate forecast skill."])


def test_only_a_hindcast_has_an_observed_track(tmp_path: Path):
    tracks: dict[RunName, pd.DataFrame] = {"baseline": synthetic_tracks()}
    divergence = compute_divergence(tracks)
    with pytest.raises(ValueError, match="a hindcast needs one"):
        export_experiment(tmp_path, hindcast_metadata(), tracks, divergence)
    with pytest.raises(ValueError, match="only a hindcast has an observed track"):
        export_experiment(
            tmp_path, example_metadata(), tracks, divergence, observed=synthetic_tracks()
        )


def test_a_forecast_comparison_has_no_observed_track(tmp_path: Path):
    manifest = example_export(tmp_path)
    assert not has_observed(tmp_path, manifest)


def invoke(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr("sys.argv", ["stormfork", *arguments])
    main()


def test_export_command_reads_the_observed_track_from_ibtracs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    write_run_directory(tmp_path / "run")
    ibtracs = write_ibtracs(tmp_path)
    monkeypatch.chdir(tmp_path)
    invoke(monkeypatch, "export", "run", "--out", "out", "--ibtracs", str(ibtracs))
    assert capsys.readouterr().out == (
        "manifest written to out/manifest.json\n"
        "release zip written to dist/atlases/hindcast_synthetic-2024100700.zip\n"
    )
    manifest = verify_manifest(tmp_path / "out")
    assert manifest.kind == "hindcast"
    assert has_observed(tmp_path / "out", manifest)


def test_export_command_needs_ibtracs_for_a_hindcast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    write_run_directory(tmp_path / "run")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "export", "run", "--out", "out")
    assert raised.value.code == 2
    assert capsys.readouterr().err == (
        "error: --ibtracs is for hindcasts, and a hindcast needs it\n"
        "next: pass the IBTrACS CSV for a hindcast and omit it otherwise\n"
    )
