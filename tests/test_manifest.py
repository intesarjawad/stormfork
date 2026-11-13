from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from pydantic import ValidationError

from stormfork.export.divergence import RunName, compute_divergence
from stormfork.export.manifest import (
    HINDCAST_LIMIT,
    ExperimentMetadata,
    Manifest,
    export_experiment,
    verify_manifest,
)
from stormfork.export.quantise import FILE_NAMES, field_metadata
from tests.test_divergence import synthetic_tracks

if TYPE_CHECKING:
    import pandas as pd


def example_metadata() -> ExperimentMetadata:
    return ExperimentMetadata.model_validate(
        {
            "schema_version": 2,
            "id": "synthetic_comparison",
            "title": "Synthetic comparison",
            "kind": "fork",
            "model": {
                "name": "fixture",
                "checkpoint": "fixture.npz",
                "md5": "0" * 32,
                "code_version": "fixture-v1",
            },
            "runtime": {"kind": "cpu", "device": "synthetic"},
            "init": {
                "source": "hres_sample",
                "time": "2024-10-07T00:00:00",
                "storm": {
                    "ibtracs_sid": "fixture",
                    "lat": 20,
                    "lon": 300,
                    "first_record_time": "2024-10-05T12:00:00",
                },
            },
            "perturbation": {"type": "relocate", "params": {"distance_km": 100}},
            "seeds": {"baseline": 0, "control": 1000},
            "members": 8,
            "steps": 4,
            "fields": field_metadata(),
            "diagnostics": {"core_land_fraction": 0, "bounds_check": "passed"},
            "limits": ["Synthetic tracks do not demonstrate forecast skill."],
            "attribution": ["Synthetic data."],
            "created": "2026-11-05T12:00:00Z",
        }
    )


def example_export(directory: Path, *, adjustment_ratio: float | None = None) -> Manifest:
    tracks: dict[RunName, pd.DataFrame] = {
        "baseline": synthetic_tracks(),
        "fork": synthetic_tracks(0.5),
        "control": synthetic_tracks(0.1),
    }
    tracks["fork"]["mslp_pa"] += tracks["fork"].step * 100
    for run in tracks:
        (directory / run).mkdir()
        for file in FILE_NAMES:
            Image.new("L", (1805, 1629), 100).save(directory / run / f"{file}.png")
    (directory / "preview").mkdir()
    Image.new("L", (360, 181), 100).save(directory / "preview/baseline_mslp_f0.png")
    return export_experiment(
        directory,
        example_metadata(),
        tracks,
        compute_divergence(tracks),
        adjustment_ratio=adjustment_ratio,
    )


def test_export_validates_and_hashes_every_evidence_format(tmp_path: Path):
    manifest = example_export(tmp_path, adjustment_ratio=1.25)
    assert verify_manifest(tmp_path) == manifest
    assert Manifest.model_validate_json((tmp_path / "manifest.json").read_bytes()) == manifest
    assert manifest.diagnostics.adjustment_ratio == 1.25
    assert manifest.diagnostics.intensity_drift_24h_hpa == 4
    assert manifest.diagnostics.core_land_fraction == 0
    assert len(manifest.runs["baseline"]) == len(FILE_NAMES) + 1
    assert set(manifest.tracks) == {"json", "geojson", "csv"}


@pytest.mark.parametrize(
    "path",
    [
        "fork/mslp.png",
        "tracks.json",
        "tracks.geojson",
        "tracks.csv",
        "divergence.json",
        "preview/baseline_mslp_f0.png",
    ],
)
def test_one_changed_byte_is_rejected(tmp_path: Path, path: str):
    example_export(tmp_path)
    artifact = tmp_path / path
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 1
    artifact.write_bytes(data)
    with pytest.raises(ValueError, match=r"SHA-256.*does not match"):
        verify_manifest(tmp_path)


def test_adjustment_ratio_can_be_absent_or_supplied_in_metadata(tmp_path: Path):
    manifest = example_export(tmp_path)
    assert manifest.diagnostics.adjustment_ratio is None
    metadata = example_metadata()
    metadata.diagnostics.adjustment_ratio = 1.1
    tracks: dict[RunName, pd.DataFrame] = {
        "baseline": synthetic_tracks(),
        "fork": synthetic_tracks(),
        "control": synthetic_tracks(),
    }
    manifest = export_experiment(tmp_path, metadata, tracks, compute_divergence(tracks))
    assert manifest.diagnostics.adjustment_ratio == metadata.diagnostics.adjustment_ratio
    with pytest.raises(ValidationError, match="finite"):
        export_experiment(
            tmp_path, metadata, tracks, compute_divergence(tracks), adjustment_ratio=float("inf")
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("limits", []),
        ("attribution", []),
        ("limits", [" "]),
        ("members", 0),
        ("kind", "unknown"),
        ("unexpected", 1),
        ("schema_version", 1),
        ("schema_version", None),
        ("parent_id", "synthetic"),
    ],
)
def test_invalid_provenance_is_rejected(key: str, value: object):
    data = example_metadata().model_dump()
    data[key] = value
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate(data)


def test_the_storm_position_is_null_only_before_its_first_record():
    data = example_metadata().model_dump()
    storm = data["init"]["storm"]
    storm.update(lat=None, lon=None)
    with pytest.raises(ValidationError, match="null exactly when the init precedes"):
        ExperimentMetadata.model_validate(data)
    storm["first_record_time"] = "2024-10-07T06:00:00"
    metadata = ExperimentMetadata.model_validate(data)
    assert (metadata.init.storm.lat, metadata.init.storm.lon) == (None, None)
    storm.update(lat=20)
    with pytest.raises(ValidationError, match="both present or both null"):
        ExperimentMetadata.model_validate(data)
    storm.update(lon=300)
    with pytest.raises(ValidationError, match="null exactly when the init precedes"):
        ExperimentMetadata.model_validate(data)


@pytest.mark.parametrize("key", ["ibtracs_sid", "first_record_time"])
def test_only_the_storm_position_may_be_null(key: str):
    data = example_metadata().model_dump()
    data["init"]["storm"][key] = None
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate(data)


def test_a_hindcast_init_is_named_by_its_parent_and_init_time():
    data = example_metadata().model_dump()
    data.update(
        kind="hindcast",
        limits=[HINDCAST_LIMIT],
        parent_id="hindcast_synthetic",
        id="hindcast_synthetic-2024100700",
    )
    assert ExperimentMetadata.model_validate(data).parent_id == "hindcast_synthetic"
    for change in ({"id": "hindcast_synthetic-2024100706"}, {"parent_id": None}):
        with pytest.raises(ValidationError, match="<parent_id>-YYYYMMDDHH"):
            ExperimentMetadata.model_validate(data | change)


def test_seed_and_field_contracts_are_validated():
    data = example_metadata().model_dump()
    data["seeds"] = {"baseline": 0, "control": 1}
    with pytest.raises(ValidationError, match="1000"):
        ExperimentMetadata.model_validate(data)
    data = example_metadata().model_dump()
    data["fields"] = field_metadata()[:-1]
    with pytest.raises(ValidationError, match="five"):
        ExperimentMetadata.model_validate(data)
    fields = field_metadata(wind_resolution_deg=2)
    data["fields"] = fields
    assert ExperimentMetadata.model_validate(data).fields[1].resolution_deg == 2
    fields[0]["range"] = [0, 1]
    with pytest.raises(ValidationError, match="quantisation"):
        ExperimentMetadata.model_validate(data)


def test_manifest_rejects_missing_runs_hashes_and_unsafe_paths(tmp_path: Path):
    manifest = example_export(tmp_path)
    data = manifest.model_dump()
    data["runs"] = {"baseline": manifest.runs["baseline"]}
    with pytest.raises(ValidationError, match="comparison runs"):
        Manifest.model_validate(data)
    data["runs"] = {"baseline": {"../outside.png": "0" * 64}, "fork": {}, "control": {}}
    with pytest.raises(ValidationError, match="relative POSIX"):
        Manifest.model_validate(data)
    data["runs"] = {"baseline": {}, "fork": {}, "control": {}}
    with pytest.raises(ValidationError, match="missing required PNG"):
        Manifest.model_validate(data)
    data["runs"] = manifest.runs
    data["tracks"] = {"json": {"path": "/outside.json", "sha256": "0" * 64}}
    with pytest.raises(ValidationError, match="relative POSIX"):
        Manifest.model_validate(data)


def test_symlink_outside_export_directory_is_rejected(tmp_path: Path):
    manifest = example_export(tmp_path)
    path = tmp_path / "tracks.csv"
    path.unlink()
    path.symlink_to(tmp_path.parent / "outside.csv")
    with pytest.raises(ValueError, match="escapes export directory"):
        verify_manifest(tmp_path)
    assert manifest.tracks["csv"].path == "tracks.csv"


@pytest.mark.parametrize("login", ["alice", "Alice-b2", "a"])
def test_run_by_accepts_github_logins(login: str):
    data = example_metadata().model_dump()
    data["run_by"] = login
    assert ExperimentMetadata.model_validate(data).run_by == login


@pytest.mark.parametrize("login", ["", "-alice", "alice-", "a--b", "a b", "a" * 40])
def test_run_by_rejects_other_text(login: str):
    data = example_metadata().model_dump()
    data["run_by"] = login
    with pytest.raises(ValidationError):
        ExperimentMetadata.model_validate(data)
