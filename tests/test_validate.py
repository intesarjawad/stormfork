import numpy as np
import pytest
import xarray as xr

from stormfork.grid import EARTH_RADIUS_KM
from stormfork.perturb.base import Taper
from stormfork.perturb.validate import PhysicalBoundsError, dry_run_statistics, validate


def weather_dataset() -> xr.Dataset:
    shape = (21, 360)
    fields = {
        "specific_humidity": 0.01,
        "mean_sea_level_pressure": 100000,
        "temperature": 290,
        "2m_temperature": 290,
        "sea_surface_temperature": 290,
        "land_sea_mask": 0,
        "geopotential_at_surface": 0,
    }
    return xr.Dataset(
        {
            name: (("lat", "lon"), np.full(shape, value, dtype=float))
            for name, value in fields.items()
        },
        coords={"lat": np.arange(-10, 11), "lon": np.arange(360)},
    )


@pytest.mark.parametrize(
    ("name", "value", "bound"),
    [
        ("specific_humidity", -0.001, ">= 0"),
        ("mean_sea_level_pressure", 86999, ">= 87000"),
        ("mean_sea_level_pressure", 108501, "<= 108500"),
        ("temperature", 169, ">= 170"),
        ("temperature", 341, "<= 340"),
        ("2m_temperature", 179, ">= 180"),
        ("2m_temperature", 341, "<= 340"),
        ("sea_surface_temperature", 269, ">= 270"),
        ("sea_surface_temperature", 311, "<= 310"),
    ],
)
def test_physical_bounds(name: str, value: float, bound: str):
    original = weather_dataset()
    changed = original.copy(deep=True)
    changed[name].values[0, 0] = value
    with pytest.raises(PhysicalBoundsError) as error:
        validate(original, changed)
    assert name in str(error.value)
    assert bound in str(error.value)
    assert str(value) in str(error.value)


@pytest.mark.parametrize(
    ("name", "lower", "upper"),
    [
        ("specific_humidity", 0, 1),
        ("mean_sea_level_pressure", 87000, 108500),
        ("temperature", 170, 340),
        ("2m_temperature", 180, 340),
        ("sea_surface_temperature", 270, 310),
    ],
)
def test_inclusive_bounds(name: str, lower: float, upper: float):
    ds = weather_dataset()
    ds[name].values[0, :2] = [lower, upper]
    validate(ds, ds.copy(deep=True))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_original_nonfinite_mask_and_new_nonfinite_in_unbounded_field(value: float):
    original = weather_dataset()
    original["u_component_of_wind"] = original.temperature * 0
    original.u_component_of_wind.values[0] = value
    changed = original.copy(deep=True)
    validate(original, changed)
    changed.u_component_of_wind.values[1, 0] = value
    with pytest.raises(PhysicalBoundsError, match="u_component_of_wind"):
        validate(original, changed)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "name",
    [
        "specific_humidity",
        "mean_sea_level_pressure",
        "temperature",
        "2m_temperature",
        "sea_surface_temperature",
    ],
)
def test_bounded_fields_reject_existing_nonfinite_values(name: str, value: float):
    original = weather_dataset()
    original[name].values[0, 0] = value
    with pytest.raises(PhysicalBoundsError, match=name):
        validate(original, original.copy(deep=True))


def test_sea_surface_temperature_may_be_undefined_only_where_there_is_land():
    original = weather_dataset()
    original.land_sea_mask.values[0, :3] = [1, 0.3, 0]
    original.sea_surface_temperature.values[0, :2] = np.nan
    validate(original, original.copy(deep=True))
    for cell, value in ((2, np.nan), (1, np.inf)):
        changed = original.copy(deep=True)
        changed.sea_surface_temperature.values[0, cell] = value
        with pytest.raises(PhysicalBoundsError, match="sea_surface_temperature: required finite"):
            validate(changed, changed.copy(deep=True))
    changed = original.copy(deep=True)
    changed.sea_surface_temperature.values[0, 2] = 269
    with pytest.raises(PhysicalBoundsError, match=">= 270"):
        validate(original, changed)


@pytest.mark.parametrize("name", ["land_sea_mask", "geopotential_at_surface"])
def test_statics_unchanged(name: str):
    original = weather_dataset()
    changed = original.copy(deep=True)
    changed[name].values[0, 0] = 1
    with pytest.raises(PhysicalBoundsError, match=rf"{name}.*unchanged.*1"):
        validate(original, changed)


def test_moved_core_land_fraction_and_strict_threshold():
    original = weather_dataset()
    original.land_sea_mask.loc[{"lon": slice(178, 182)}] = 0.2
    assert validate(original, original, moved_taper=Taper(0, 0, 400)) == 0
    with pytest.raises(PhysicalBoundsError, match=r"land_sea_mask.*< 0.2.*0.2"):
        validate(original, original, moved_taper=Taper(0, 180, 400))
    original.land_sea_mask.loc[{"lon": slice(178, 182)}] = 0.19
    assert validate(original, original, moved_taper=Taper(0, 180, 400)) == pytest.approx(0.19)
    assert validate(original, original) is None


def test_core_must_have_known_land_coverage():
    original = weather_dataset()
    with pytest.raises(ValueError, match=r"core.*grid"):
        validate(original, original, moved_taper=Taper(0.5, 0.5, 1))
    original.land_sea_mask.values[10, 0] = np.nan
    with pytest.raises(PhysicalBoundsError, match=r"land_sea_mask.*finite"):
        validate(original, original, moved_taper=Taper(0, 0, 400))


def test_dry_run_statistics():
    original = weather_dataset()
    changed = original.copy(deep=True)
    changed.temperature[:] += 2
    changed.sea_surface_temperature[:] = np.nan
    original.sea_surface_temperature[:] = np.nan
    output = dry_run_statistics(original, changed, Taper(0, 0, 400))
    assert "temperature: min=2 max=2 mean=2" in output
    assert "sea_surface_temperature: min=nan max=nan mean=nan" in output
    assert "core land fraction: 0" in output
    area_line = next(line for line in output.splitlines() if line.startswith("taper area:"))
    area = float(area_line.split()[2])
    cap_area = 2 * np.pi * EARTH_RADIUS_KM**2 * (1 - np.cos(400 / EARTH_RADIUS_KM))
    assert area == pytest.approx(cap_area, rel=0.15)
    assert "km²" in area_line
    assert changed.temperature.values[0, 0] == 292


def test_custom_names_and_dimension_order():
    original = weather_dataset().rename({"temperature": "t", "land_sea_mask": "lsm"})
    changed = original.transpose("lon", "lat").copy(deep=True)
    names = {"temperature": "t", "land_sea_mask": "lsm"}
    validate(original, changed, moved_taper=Taper(0, 0, 400), names=names)
    assert "t: min=0 max=0 mean=0" in dry_run_statistics(
        original, changed, Taper(0, 0, 400), names=names
    )
    changed.t.values[0, 0] = 400
    with pytest.raises(PhysicalBoundsError, match=r"t.*<= 340.*400"):
        validate(original, changed, names=names)


@pytest.mark.parametrize("operation", ["coordinates", "variables", "dimensions"])
def test_incompatible_layout(operation: str):
    original = weather_dataset()
    if operation == "coordinates":
        changed = original.copy(deep=True)
        changed["lat"] = original.lat + 1
    elif operation == "variables":
        changed = original.drop_vars("temperature")
    else:
        changed = original.copy(deep=True)
        changed["temperature"] = changed.temperature.isel(lat=0)
    with pytest.raises(ValueError, match=r"must (match|contain)"):
        validate(original, changed)
    with pytest.raises(ValueError, match=r"must (match|contain)"):
        dry_run_statistics(original, changed, Taper(0, 0, 400))


def test_all_frames_and_levels_checked():
    original = weather_dataset().expand_dims(time=[0, 1], level=[1000, 925]).copy(deep=True)
    changed = original.copy(deep=True)
    changed.temperature.values[1, 1, 0, 0] = 341
    with pytest.raises(PhysicalBoundsError, match=r"temperature.*<= 340.*341"):
        validate(original, changed)
