"""The edit, ratio and sweep-stop logic every forecast run uses, without the model."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from stormfork.errors import UserError
from stormfork.grid import LAT, LON
from stormfork.perturb import Taper, adjustment_ratio
from stormfork.perturb.base import MIN_RADIUS_KM
from stormfork.run import forks
from stormfork.scenarios import Scenario
from stormfork.track import StormStart

INIT = "2024-10-07T00:00"
MSLP = "mean_sea_level_pressure"
SIX_HOURS = np.timedelta64(6, "h").astype("timedelta64[ns]")
FIRST_RECORD = datetime.fromisoformat("2024-10-05T12:00")
GALE_RADII_NM = (200.0, float("nan"), 150.0, 100.0)
STORM = StormStart((22.5, 266.6), FIRST_RECORD, GALE_RADII_NM)
BEFORE_GENESIS = StormStart(None, FIRST_RECORD)


def scenario(**changes: object) -> Scenario:
    recipe: dict[str, object] = {
        "id": "x",
        "kind": "fork",
        "storm": "s",
        "init": INIT,
        "source": "hres_sample",
        "edit": {"type": "noise", "params": {"amplitude": 0.01, "seed": 0}},
        "seeds": {"baseline": 0, "control": 1000},
    }
    return Scenario.model_validate(recipe | changes)


def test_an_edit_parameter_the_recipe_leaves_out_is_named():
    recipe = scenario(
        edit={"type": "scale_intensity", "params": {}},
        taper={"centre": [22.5, -93.4], "radius_km": 400.0},
    )
    with pytest.raises(UserError, match="scale_intensity needs a number for factor") as raised:
        forks.apply_edit(recipe, xr.Dataset())
    assert raised.value.next_step == "set edit.params.factor in the recipe"


def first_step(initial_pa: float, first_pa: float) -> xr.Dataset:
    """Member 0's uniform pressure at the init and after one step."""
    pressure = np.empty((2, LAT.size, LON.size))
    pressure[0], pressure[1] = initial_pa, first_pa
    return xr.Dataset(
        {MSLP: (("time", "lat", "lon"), pressure)},
        coords={"time": np.array([0, 1]) * SIX_HOURS, "lat": LAT, "lon": LON},
    )


def test_first_step_pressure_reads_member_zero_at_the_init_and_after_one_step():
    grid = {"lat": LAT, "lon": LON}
    inputs = xr.Dataset(
        {MSLP: (("batch", "time", "lat", "lon"), np.empty((1, 2, LAT.size, LON.size)))},
        coords={"time": np.array([-1, 0]) * SIX_HOURS, **grid},
    )
    inputs[MSLP][:, 0] = 90000.0
    inputs[MSLP][:, 1] = 101000.0
    forecast = xr.Dataset(
        {
            MSLP: (
                ("member", "batch", "time", "lat", "lon"),
                np.empty((2, 1, 1, LAT.size, LON.size)),
            )
        },
        coords={"time": [SIX_HOURS], **grid},
    )
    forecast[MSLP][0] = 101100.0
    forecast[MSLP][1] = 99000.0
    pressure = forks.first_step_pressure(inputs, forecast)
    assert pressure[MSLP].isel(batch=0, lat=0, lon=0).to_numpy().tolist() == [101000.0, 101100.0]


def test_the_adjustment_ratio_compares_first_steps_in_the_taper_annulus():
    recipe = scenario(
        edit={"type": "scale_intensity", "params": {"factor": 0.8}},
        taper={"centre": [22.5, -93.4], "radius_km": 400.0},
    )
    ratio = forks.first_step_adjustment_ratio(
        recipe, BEFORE_GENESIS, first_step(100800.0, 101100.0), first_step(101000.0, 101100.0)
    )
    assert ratio == pytest.approx((101100 - 100800) / (101100 - 101000))


def test_a_relocation_takes_the_larger_ratio_of_its_original_and_moved_annulus():
    recipe = scenario(
        edit={"type": "relocate", "params": {"east_km": 600, "north_km": 0}},
        taper={"centre": [22.5, -93.4], "radius_km": 400.0},
    )
    moved = forks.moved_taper(recipe)
    assert moved is not None
    baseline, fork = first_step(101000.0, 101100.0), first_step(101000.0, 101100.0)
    # Only the moved annulus reaches east of 271 E, where the fork changes three times as much.
    fork[MSLP][1][:, LON >= 271] = 101300.0
    at_original = adjustment_ratio(fork, baseline, Taper(22.5, -93.4, 400.0))
    assert at_original == pytest.approx(1)
    at_moved = adjustment_ratio(fork, baseline, moved)
    assert at_moved > 1
    assert forks.first_step_adjustment_ratio(recipe, BEFORE_GENESIS, fork, baseline) == at_moved


def test_an_edit_without_a_taper_is_measured_in_the_taper_the_storm_would_get():
    fork, baseline = first_step(100800.0, 101100.0), first_step(101000.0, 101100.0)
    assert forks.first_step_adjustment_ratio(scenario(), STORM, fork, baseline) == pytest.approx(3)
    row = dict(
        zip(("USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"), GALE_RADII_NM, strict=True)
    )
    tapered = Taper.from_ibtracs({**row, "USA_LAT": 22.5, "USA_LON": 266.6})
    assert tapered.radius_km == pytest.approx(1.5 * 200 * 1.852)
    assert forks.ratio_tapers(scenario(), STORM) == [tapered]


def test_a_storm_without_a_gale_radius_is_measured_at_the_smallest_taper():
    unrecorded = STORM._replace(gale_radii_nm=(float("nan"),) * 4)
    assert forks.ratio_tapers(scenario(), unrecorded) == [Taper(22.5, 266.6, MIN_RADIUS_KM)]
    assert forks.ratio_tapers(scenario(), STORM._replace(gale_radii_nm=())) == [
        Taper(22.5, 266.6, MIN_RADIUS_KM)
    ]


def test_an_edit_without_a_taper_is_not_measured_before_genesis():
    fork, baseline = first_step(100800.0, 101100.0), first_step(101000.0, 101100.0)
    assert forks.ratio_tapers(scenario(), BEFORE_GENESIS) == []
    assert forks.first_step_adjustment_ratio(scenario(), BEFORE_GENESIS, fork, baseline) is None


def test_a_sweep_expands_to_its_null_fork_and_one_stop_per_other_value():
    sweep = scenario(
        kind="sweep",
        edit={"type": "relocate", "params": {"north_km": 0}},
        sweep={"param": "east_km", "values": [-300, 0, 150.5]},
        taper={"centre": [24.8, -75.9], "radius_km": 833.4},
    )
    recipes = forks.fork_recipes(sweep)
    assert list(recipes) == ["stops/east_km_-300", "fork", "stops/east_km_150.5"]
    assert {run: recipe.kind for run, recipe in recipes.items()} == dict.fromkeys(recipes, "fork")
    assert [recipe.edit.params if recipe.edit else None for recipe in recipes.values()] == [
        {"north_km": 0, "east_km": -300},
        {"north_km": 0, "east_km": 0},
        {"north_km": 0, "east_km": 150.5},
    ]
    assert forks.fork_recipes(scenario()) == {"fork": scenario()}
    assert forks.fork_recipes(scenario(kind="hindcast", edit=None)) == {}
