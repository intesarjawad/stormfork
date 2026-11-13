from itertools import product
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pytest
import xarray as xr

from tests.test_cli import invoke, write_netcdf


class ZarrWritable(Protocol):
    def to_zarr(
        self,
        store: str,
        *,
        mode: Literal["w"],
        consolidated: bool,
        zarr_format: int,
    ) -> object: ...


def write_zarr(dataset: ZarrWritable, path: Path) -> None:
    dataset.to_zarr(str(path), mode="w", consolidated=True, zarr_format=2)


@pytest.fixture
def state() -> xr.Dataset:
    lat, lon = np.arange(-10, 11), np.arange(360)
    low = np.exp(-(lat[:, None] ** 2 + (lon - 180) ** 2) / 2)
    return xr.Dataset(
        {
            "mean_sea_level_pressure": (("lat", "lon"), 101000 - 3000 * low),
            "sea_surface_temperature": (("lat", "lon"), np.full(low.shape, 290.0)),
            "land_sea_mask": (("lat", "lon"), np.zeros_like(low)),
        },
        coords={"lat": lat, "lon": lon},
    )


@pytest.mark.parametrize(
    "case",
    list(
        product(
            [False, True],
            [
                ("relocate", "--east-km", "111.2"),
                ("scale_intensity", "--factor", "1.1"),
                ("noise", "--amplitude", "0.01", "--seed", "42"),
            ],
        )
    ),
)
def test_each_atmospheric_edit_changes_pressure_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    state: xr.Dataset,
    case: tuple[bool, tuple[str, ...]],
):
    zarr, options = case
    source = tmp_path / ("state.zarr" if zarr else "state.nc")
    if zarr:
        write_zarr(state, source)
    else:
        write_netcdf(state, source)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    arguments = (
        "perturb",
        str(source),
        "--dry-run",
        "--centre-lat",
        "0",
        "--centre-lon",
        "180",
        "--radius-km",
        "400",
        "--operation",
        *options,
    )
    invoke(monkeypatch, *arguments)
    first = capsys.readouterr().out
    invoke(monkeypatch, *arguments)
    assert capsys.readouterr().out == first
    pressure = next(
        line for line in first.splitlines() if line.startswith("mean_sea_level_pressure:")
    )
    assert pressure.startswith("mean_sea_level_pressure: min=-")
    assert "land_sea_mask: min=0 max=0 mean=0" in first
    assert max(map(len, first.splitlines())) <= 80
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "options",
    [("scale_intensity",), ("sst_anomaly",), ("noise",)],
)
def test_edit_requires_its_own_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    state: xr.Dataset,
    options: tuple[str, ...],
):
    source = tmp_path / "state.nc"
    write_netcdf(state, source)
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            "perturb",
            str(source),
            "--dry-run",
            "--centre-lat",
            "0",
            "--centre-lon",
            "180",
            "--radius-km",
            "400",
            "--operation",
            *options,
        )
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"error: {options[0]} requires --")
    assert len(captured.err.splitlines()) == 2


def test_physical_validation_failure_precedes_statistics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    state: xr.Dataset,
):
    state["sea_surface_temperature"] += 20
    source = tmp_path / "state.nc"
    write_netcdf(state, source)
    with pytest.raises(SystemExit) as raised:
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
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "error: sea_surface_temperature: required <= 310; extreme=311\n"
        "next: check the reported data or path and retry\n"
    )


def test_relocation_validates_destination_land_overlap(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    state: xr.Dataset,
):
    state["land_sea_mask"].values[:] = np.broadcast_to(
        (state.lon.values >= 183).astype(float), state.land_sea_mask.shape
    )
    source = tmp_path / "state.nc"
    write_netcdf(state, source)
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            "perturb",
            str(source),
            "--dry-run",
            "--operation",
            "relocate",
            "--centre-lat",
            "0",
            "--centre-lon",
            "180",
            "--radius-km",
            "400",
            "--east-km",
            "500",
        )
    assert raised.value.code == 1
    assert "core land fraction must be < 0.2" in capsys.readouterr().err
