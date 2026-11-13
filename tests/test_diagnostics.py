import numpy as np
import pytest
import xarray as xr

from stormfork.grid import Grid, great_circle_km
from stormfork.perturb.base import Taper
from stormfork.perturb.diagnostics import DiagnosticError, adjustment_ratio

TAPER = Taper(0, 0, 400)


def forecast(tendency: float = 2.0) -> xr.Dataset:
    pressure = np.full((3, 9, 360), 100000.0)
    pressure[1] += tendency
    pressure[2] += 1000
    return xr.Dataset(
        {"mean_sea_level_pressure": (("time", "lat", "lon"), pressure)},
        coords={
            "time": np.array(
                ["2012-10-26T00", "2012-10-26T06", "2012-10-26T12"], dtype="datetime64[ns]"
            ),
            "lat": np.arange(-4, 5),
            "lon": np.arange(360),
        },
    )


def test_ratio_uses_rms_in_annulus_and_first_two_steps():
    baseline, fork = forecast(), forecast(6)
    grid = Grid.from_dataset(fork)
    distance = great_circle_km(0, 0, grid.lat[:, None], grid.lon)
    annulus = (distance >= 200) & (distance <= 400)
    baseline.mean_sea_level_pressure.values[1, 4, 2] += 2
    fork.mean_sea_level_pressure.values[1, 4, 2] -= 14
    fork.mean_sea_level_pressure.values[1][~annulus] += 500
    expected = np.sqrt((36 * (annulus.sum() - 1) + 64) / (4 * (annulus.sum() - 1) + 16))
    before = fork.copy(deep=True)
    assert adjustment_ratio(fork, baseline, TAPER) == pytest.approx(expected)
    assert fork.identical(before)


@pytest.mark.parametrize("factor", [0.0, 1.0, -3.0])
def test_uniform_tendencies(factor: float):
    assert adjustment_ratio(forecast(2 * factor), forecast(), TAPER) == pytest.approx(abs(factor))


@pytest.mark.parametrize("boundary", ["inner", "outer"])
def test_annulus_includes_boundaries(boundary: str):
    baseline, fork = forecast(), forecast()
    distance = float(great_circle_km(0, 0, 0, 2))
    taper = Taper(0, 0, distance * (2 if boundary == "inner" else 1))
    fork.mean_sea_level_pressure.values[1, 4, 2] += 20
    assert adjustment_ratio(fork, baseline, taper) > 1


def test_member_zero_custom_names_and_dimension_order():
    baseline = forecast().expand_dims(member=[7, 9]).copy(deep=True)
    fork = forecast(6).expand_dims(member=[7, 9]).copy(deep=True)
    baseline.mean_sea_level_pressure.values[1, 1] = np.nan
    fork.mean_sea_level_pressure.values[1, 1] += 500
    names = {"mean_sea_level_pressure": "mslp"}
    baseline = baseline.rename(names).transpose("lon", "time", "member", "lat")
    fork = fork.rename(names).transpose("lat", "member", "lon", "time")
    assert adjustment_ratio(fork, baseline, TAPER, names=names) == pytest.approx(3)


def test_unsigned_pressure_does_not_wrap():
    baseline, fork = forecast(-2), forecast(6)
    for ds in (baseline, fork):
        pressure = ds.mean_sea_level_pressure
        ds["mean_sea_level_pressure"] = pressure.copy(data=pressure.values.astype(np.uint32))
    assert adjustment_ratio(fork, baseline, TAPER) == pytest.approx(3)


def test_zero_baseline_tendency():
    with pytest.raises(DiagnosticError, match=r"baseline.*zero"):
        adjustment_ratio(forecast(), forecast(0), TAPER)


def test_empty_annulus():
    with pytest.raises(DiagnosticError, match=r"annulus.*no grid cell"):
        adjustment_ratio(forecast(), forecast(), Taper(0, 0, 1))


@pytest.mark.parametrize("source", ["fork", "baseline"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_annulus_tendency(source: str, value: float):
    inputs = {"fork": forecast(), "baseline": forecast()}
    inputs[source].mean_sea_level_pressure.values[1, 4, 2] = value
    with pytest.raises(DiagnosticError, match=rf"{source}.*finite"):
        adjustment_ratio(inputs["fork"], inputs["baseline"], TAPER)


@pytest.mark.parametrize("source", ["fork", "baseline"])
@pytest.mark.parametrize("invalid", ["missing", "time", "dimensions", "member"])
def test_invalid_pressure_layout(source: str, invalid: str):
    ds = forecast()
    if invalid == "missing":
        ds = ds.drop_vars("mean_sea_level_pressure")
    elif invalid == "time":
        ds = ds.isel(time=slice(0, 1))
    elif invalid == "member":
        ds = ds.expand_dims(member=[])
    else:
        ds = ds.expand_dims(level=[1000])
    inputs = {"fork": forecast(), "baseline": forecast()}
    inputs[source] = ds
    with pytest.raises(DiagnosticError, match=source):
        adjustment_ratio(inputs["fork"], inputs["baseline"], TAPER)


def test_grids_must_match():
    baseline = forecast()
    baseline["lat"] = np.arange(-3, 6)
    with pytest.raises(ValueError, match=r"lat.*lon.*match"):
        adjustment_ratio(forecast(), baseline, TAPER)


def test_invalid_grid():
    with pytest.raises(ValueError, match="lon"):
        adjustment_ratio(forecast().isel(lon=slice(1, None)), forecast(), TAPER)
