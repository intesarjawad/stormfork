from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PIL import Image

from stormfork.cli import main
from stormfork.export.manifest import verify_manifest
from stormfork.export.quantise import FILE_NAMES
from tests.test_divergence import synthetic_tracks
from tests.test_manifest import example_export, example_metadata
from tests.test_tracker import gaussian_fields, ibtracs_dataset


class NetcdfWritable(Protocol):
    def to_netcdf(self, path: Path, *, engine: Literal["scipy"]) -> None: ...


def write_netcdf(dataset: NetcdfWritable, path: Path) -> None:
    dataset.to_netcdf(path, engine="scipy")


def invoke(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr("sys.argv", ["stormfork", *arguments])
    main()


def test_taper_from_retained_ibtracs_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    invoke(
        monkeypatch,
        "scenarios",
        "taper",
        "2005236N23285",
        "--ibtracs",
        "tests/fixtures/ibtracs_katrina_taper.csv",
        "--init",
        "2005-08-24T12:00:00",
    )
    assert capsys.readouterr().out == (
        "taper:\n  centre_lat: 24.5\n  centre_lon: -76.5\n  radius_km: 400.0\n"
    )


def test_perturb_dry_run_preserves_input_and_prints_statistics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    dataset = xr.Dataset(
        {
            "sea_surface_temperature": (("lat", "lon"), np.full((21, 360), 290.0)),
            "land_sea_mask": (("lat", "lon"), np.zeros((21, 360))),
        },
        coords={"lat": np.arange(-10, 11), "lon": np.arange(360)},
    )
    source = tmp_path / "state.nc"
    write_netcdf(dataset, source)
    before = source.read_bytes()
    invoke(
        monkeypatch,
        "perturb",
        str(source),
        "--dry-run",
        "--operation",
        "sst_anomaly",
        "--centre-lat",
        "0",
        "--centre-lon",
        "180",
        "--radius-km",
        "400",
        "--sigma-km",
        "200",
        "--delta-k",
        "1",
    )
    output = capsys.readouterr().out
    assert "sea_surface_temperature: min=0 max=1 mean=" in output
    assert "land_sea_mask: min=0 max=0 mean=0" in output
    assert "core land fraction: 0" in output
    assert "taper area:" in output
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]


def test_track_supplied_datasets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    fields, observations, output = (tmp_path / name for name in ("fields.nc", "obs.nc", "out.csv"))
    write_netcdf(gaussian_fields(steps=3), fields)
    write_netcdf(ibtracs_dataset(), observations)
    invoke(
        monkeypatch,
        "track",
        str(fields),
        "--ibtracs",
        str(observations),
        "--init",
        "2024-10-07T00:00:00",
        "--out",
        str(output),
    )
    result = pd.read_csv(output)
    assert result.step.tolist() == [0, 1, 2]
    assert result.track_id.tolist() == ["target"] * 3
    assert np.allclose(result.mslp_pa, 98000)
    assert np.allclose(result.lon, 180, atol=0.01)
    assert "3 track samples" in capsys.readouterr().out


def test_export_run_directory_matches_the_direct_export(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    direct = tmp_path / "direct"
    direct.mkdir()
    example_export(direct)
    run_directory = tmp_path / "run"
    for run, tracks in (
        ("baseline", synthetic_tracks()),
        ("fork", synthetic_tracks(0.5)),
        ("control", synthetic_tracks(0.1)),
    ):
        (run_directory / run).mkdir(parents=True)
        if run == "fork":
            tracks["mslp_pa"] += tracks.step * 100
        tracks.to_csv(run_directory / run / "tracks.csv", index=False)
        for file in FILE_NAMES:
            Image.new("L", (1805, 1629), 100).save(run_directory / run / f"{file}.png")
    (run_directory / "preview").mkdir()
    Image.new("L", (360, 181), 100).save(run_directory / "preview/baseline_mslp_f0.png")
    (run_directory / "metadata.json").write_text(example_metadata().model_dump_json())
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    invoke(monkeypatch, "export", "run", "--out", "out")
    assert capsys.readouterr().out == (
        "manifest written to out/manifest.json\n"
        "release zip written to dist/atlases/synthetic_comparison.zip\n"
    )
    for name in ("manifest.json", "tracks.json", "tracks.csv", "tracks.geojson", "divergence.json"):
        assert (out / name).read_bytes() == (direct / name).read_bytes()
    assert verify_manifest(out).id == "synthetic_comparison"
