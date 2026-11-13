from collections.abc import Callable, Mapping

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from stormfork.grid import great_circle_km
from stormfork.perturb import Taper, adjustment_ratio, noise, relocate, scale_intensity, sst_anomaly
from stormfork.perturb.diagnostics import DiagnosticError
from stormfork.perturb.validate import dry_run_statistics, validate

TAPER = Taper(0, 0, 400)
OPERATIONS = ("relocate", "scale_intensity", "sst_anomaly", "noise", "validate", "statistics")


def dataset() -> xr.Dataset:
    shape = (2, 2, 21, 360)
    return xr.Dataset(
        {
            "temperature": (("time", "level", "lat", "lon"), np.full(shape, 290.0)),
            "sea_surface_temperature": (("lat", "lon"), np.full(shape[-2:], 290.0)),
            "land_sea_mask": (("lat", "lon"), np.zeros(shape[-2:])),
        },
        coords={
            "time": [0, 1],
            "level": [1000, 925],
            "lat": np.arange(-10, 11),
            "lon": np.arange(360),
        },
    )


def apply(operation: str, ds: xr.Dataset, names: Mapping[str, str], *, identity: bool) -> object:
    match operation:
        case "relocate":
            return relocate(ds, TAPER, east_km=0 if identity else 100, names=names)
        case "scale_intensity":
            return scale_intensity(ds, TAPER, factor=1 if identity else 1.1, names=names)
        case "sst_anomaly":
            return sst_anomaly(
                ds, centre=(0, 0), sigma_km=400, delta_k=0 if identity else 1, names=names
            )
        case "noise":
            return noise(ds, amplitude=0 if identity else 0.01, seed=42, names=names)
        case "validate":
            return validate(ds, ds, names=names)
        case "statistics":
            return dry_run_statistics(ds, ds, TAPER, names=names)
        case _:
            raise AssertionError(operation)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("mapping", [{"typo": "temperature"}, {"temperature": "absent"}])
def test_explicit_names_fail_loudly(operation: str, mapping: dict[str, str], *, identity: bool):
    error, message = (ValueError, "typo") if "typo" in mapping else (KeyError, "absent")
    with pytest.raises(error, match=message):
        apply(operation, dataset(), mapping, identity=identity)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_absent_default_atmospheric_fields_are_optional(operation: str):
    apply(operation, dataset().drop_vars("temperature"), {}, identity=False)


@pytest.mark.parametrize("operation", OPERATIONS[:4])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_edits_require_floating_fields(operation: str, dtype: type[np.integer], *, identity: bool):
    ds = dataset()
    ds["temperature"] = ds.temperature.astype(dtype)
    with pytest.raises(TypeError, match=r"temperature.*floating"):
        apply(operation, ds, {}, identity=identity)


@pytest.mark.parametrize("levels", [[850, 500], [1000], [925]])
@pytest.mark.parametrize("delta_k", [0, 1])
def test_sst_requires_both_response_levels(levels: list[int], delta_k: float):
    ds = dataset().reindex(level=levels)
    with pytest.raises(KeyError, match=r"temperature.*(1000|925)"):
        sst_anomaly(ds, centre=(0, 0), sigma_km=400, delta_k=delta_k)


def forecast(interval_hours: int = 6) -> xr.Dataset:
    ds = dataset().drop_vars(("temperature", "sea_surface_temperature", "level"))
    pressure = np.full((2, 21, 360), 100000.0)
    pressure[1] += 2
    ds["mean_sea_level_pressure"] = (("time", "lat", "lon"), pressure)
    ds["time"] = np.array([0, interval_hours], dtype="timedelta64[h]")
    return ds


def test_diagnostic_squeezes_singleton_batch_with_member_and_permuted_axes():
    ds = forecast()
    wrapped = ds.expand_dims(batch=[0], member=[0, 1]).transpose(
        "lon", "batch", "time", "lat", "member"
    )
    assert adjustment_ratio(wrapped, ds, TAPER) == pytest.approx(1)
    assert adjustment_ratio(ds, wrapped, TAPER) == pytest.approx(1)


@pytest.mark.parametrize("source", ["fork", "baseline"])
@pytest.mark.parametrize("batch", [[], [0, 1]])
def test_diagnostic_rejects_nonsingleton_batch(source: str, batch: list[int]):
    inputs = {"fork": forecast(), "baseline": forecast()}
    inputs[source] = inputs[source].expand_dims(batch=batch)
    with pytest.raises(DiagnosticError, match=rf"{source}.*batch"):
        adjustment_ratio(inputs["fork"], inputs["baseline"], TAPER)


@pytest.mark.parametrize(
    ("fork_hours", "baseline_hours"), [(12, 6), (6, 12), (12, 12), (0, 6), (-6, 6)]
)
def test_diagnostic_requires_six_hour_interval(fork_hours: int, baseline_hours: int):
    with pytest.raises(DiagnosticError, match=r"interval.*six.hour"):
        adjustment_ratio(forecast(fork_hours), forecast(baseline_hours), TAPER)


@pytest.mark.parametrize("source", ["fork", "baseline", "both"])
def test_diagnostic_allows_absent_time_coordinate(source: str):
    inputs = {"fork": forecast(), "baseline": forecast()}
    for name, ds in inputs.items():
        if source in (name, "both"):
            inputs[name] = ds.drop_vars("time")
    assert adjustment_ratio(inputs["fork"], inputs["baseline"], TAPER) == pytest.approx(1)


@pytest.mark.parametrize(
    "mapping", [{"typo": "mean_sea_level_pressure"}, {"mean_sea_level_pressure": "absent"}]
)
def test_diagnostic_explicit_names_fail_loudly(mapping: dict[str, str]):
    error, message = (ValueError, "typo") if "typo" in mapping else (KeyError, "absent")
    with pytest.raises(error, match=message):
        adjustment_ratio(forecast(), forecast(), TAPER, names=mapping)


@pytest.mark.parametrize("name", ["sea_surface_temperature", "2m_temperature", "temperature"])
def test_sst_rejects_integer_temperature_fields_with_custom_names(name: str):
    ds = dataset()
    ds["2m_temperature"] = ds.sea_surface_temperature.copy()
    ds[name] = ds[name].copy(data=ds[name].values.astype(np.int64))
    ds = ds.rename({name: "custom"})
    with pytest.raises(TypeError, match=r"custom.*floating"):
        sst_anomaly(ds, centre=(0, 0), sigma_km=400, delta_k=1, names={name: "custom"})


def test_sst_requires_pressure_coordinate_on_temperature():
    ds = dataset().isel(level=0, drop=True)
    with pytest.raises(KeyError, match=r"temperature.*pressure levels"):
        sst_anomaly(ds, centre=(0, 0), sigma_km=400, delta_k=1)


@pytest.mark.parametrize("source", ["fork", "baseline"])
def test_diagnostic_mapped_pressure_must_exist_in_each_input(source: str):
    inputs = {"fork": forecast(), "baseline": forecast()}
    inputs[source] = inputs[source].rename({"mean_sea_level_pressure": "other"})
    with pytest.raises(KeyError, match="mean_sea_level_pressure"):
        adjustment_ratio(
            inputs["fork"],
            inputs["baseline"],
            TAPER,
            names={"mean_sea_level_pressure": "mean_sea_level_pressure"},
        )


@pytest.mark.parametrize("source", ["fork", "baseline"])
@pytest.mark.parametrize(
    "time", [np.array([0, 6]), np.array(["NaT", "NaT"], dtype="datetime64[ns]")]
)
def test_diagnostic_rejects_unmeasurable_interval(source: str, time: NDArray[np.generic]):
    inputs = {"fork": forecast(), "baseline": forecast()}
    inputs[source]["time"] = time
    with pytest.raises(DiagnosticError, match=r"interval.*six.hour"):
        adjustment_ratio(inputs["fork"], inputs["baseline"], TAPER)


def moist_core() -> xr.Dataset:
    """A nearly dry atmosphere around one sharp moist core, as in the tropics aloft."""
    lat, lon = np.arange(0, 41.0), np.arange(0, 360.0)
    distance = great_circle_km(20, 280, lat[:, None], lon)
    humidity = 1e-7 + 0.02 * np.exp(-0.5 * (distance / 200) ** 2)
    return xr.Dataset(
        {"specific_humidity": (("lat", "lon"), humidity)}, coords={"lat": lat, "lon": lon}
    )


HUMIDITY_EDITS: dict[str, Callable[[xr.Dataset], xr.Dataset]] = {
    "relocate": lambda ds: relocate(ds, Taper(20, 280, 900), east_km=300),
    "scale_intensity": lambda ds: scale_intensity(ds, Taper(20, 280, 900), factor=1.5),
    "noise": lambda ds: noise(ds, amplitude=0.01, seed=0),
}


@pytest.mark.parametrize("edit", HUMIDITY_EDITS)
def test_edits_keep_specific_humidity_nonnegative(edit: str):
    original = moist_core()
    edited = HUMIDITY_EDITS[edit](original)
    assert float(edited.specific_humidity.min()) == 0
    assert np.count_nonzero(edited.specific_humidity != original.specific_humidity) > 1
    validate(original, edited)
