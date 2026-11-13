import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stormfork.track.transition import continue_through_transition

INIT_TIME = np.datetime64("2024-10-07T00:00:00", "ns")
STEPS = 12


def low_fields(
    centres: list[list[tuple[float, float]]], *, depth: float = 2000.0, width: float = 4.0
) -> xr.Dataset:
    """Fields whose mean sea level pressure has Gaussian lows; centres[step] lists (lat, lon)."""
    lat, lon = np.arange(-40.0, 61.0), np.arange(360.0)
    pressure = np.full((1, STEPS, lat.size, lon.size), 101000.0)
    for step, lows in enumerate(centres):
        for low_lat, low_lon in lows:
            across = (lon - low_lon + 180) % 360 - 180
            distance = (lat[:, None] - low_lat) ** 2 + across**2
            pressure[0, step] -= depth * np.exp(-distance / (2 * width**2))
    dims = ("member", "time", "lat", "lon")
    return xr.Dataset(
        {
            "mean_sea_level_pressure": (dims, pressure),
            "cyclone_usa_pres_disc": (dims, np.full_like(pressure, 990.0)),
            "cyclone_usa_wind_disc": (dims, np.full_like(pressure, 50.0)),
        },
        coords={
            "member": [0],
            "time": np.arange(STEPS) * np.timedelta64(6, "h") + np.timedelta64(6, "h"),
            "lat": lat,
            "lon": lon,
        },
    )


def matched_until(last_step: int, centres: list[list[tuple[float, float]]]) -> pd.DataFrame:
    steps = np.arange(1, last_step + 1)
    return pd.DataFrame(
        {
            "member": 0,
            "step": steps,
            "valid_time": pd.Timestamp(INIT_TIME) + pd.to_timedelta(steps * 6, unit="h"),
            "lat": [centres[s - 1][0][0] for s in steps],
            "lon": [centres[s - 1][0][1] for s in steps],
            "mslp_pa": 98000.0,
            "wind_ms": 40.0,
            "track_id": "storm",
        }
    )


def eastward(
    start: float, *, steps: int = STEPS, lat: float = 35.0
) -> list[list[tuple[float, float]]]:
    return [[(lat, float((start + 2 * step) % 360))] for step in range(steps)]


def expected_lon(centres: list[list[tuple[float, float]]], steps: pd.Series) -> np.ndarray:
    return np.array([centres[int(step) - 1][0][1] for step in steps])


def test_a_track_continues_with_a_moving_low_to_the_last_step():
    centres = eastward(280)
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    assert result.step.tolist() == list(range(1, STEPS + 1))
    assert result.track_id.unique().tolist() == ["storm"]
    added = result[result.step > 4]
    assert np.allclose(added.lat, 35)
    assert np.allclose(added.lon, expected_lon(centres, added.step))
    assert np.allclose(added.mslp_pa, 99000)
    assert np.allclose(added.wind_ms, 50 * 1852 / 3600)
    assert added.valid_time.tolist() == [
        pd.Timestamp(INIT_TIME) + pd.Timedelta(hours=6 * int(s)) for s in added.step
    ]
    pd.testing.assert_frame_equal(result[result.step <= 4], matched_until(4, centres))


def test_the_track_ends_when_the_low_fills():
    centres = eastward(280, steps=7) + [[] for _ in range(STEPS - 7)]
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    assert result.step.max() == 7


def test_a_shallow_trough_is_not_a_closed_low():
    centres = eastward(280)
    fields = low_fields(centres, depth=100.0)
    matched = matched_until(4, centres)
    assert continue_through_transition(matched, fields, INIT_TIME).step.max() == 4


def test_the_low_nearest_the_extrapolated_position_is_followed():
    centres = [[*one, (38.0, one[0][1])] for one in eastward(280)]
    fields = low_fields(centres, width=1.5)
    result = continue_through_transition(matched_until(4, centres), fields, INIT_TIME)
    assert np.allclose(result.lon, expected_lon(centres, result.step))


def test_a_low_farther_than_the_search_radius_is_not_followed():
    centres = eastward(280)
    jumped = [*centres[:4], *[[(lat, lon + 8)] for [(lat, lon)] in centres[4:]]]
    result = continue_through_transition(matched_until(4, centres), low_fields(jumped), INIT_TIME)
    assert result.step.max() == 4


def test_a_track_crosses_the_longitude_seam():
    centres = eastward(350)
    result = continue_through_transition(matched_until(3, centres), low_fields(centres), INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result.lon, expected_lon(centres, result.step))


def test_a_track_already_at_the_last_step_is_unchanged():
    centres = eastward(280)
    matched = matched_until(STEPS, centres)
    pd.testing.assert_frame_equal(
        continue_through_transition(matched, low_fields(centres), INIT_TIME), matched
    )


def test_fields_without_pressure_name_the_missing_variable():
    centres = eastward(280)
    fields = low_fields(centres).drop_vars("mean_sea_level_pressure")
    with pytest.raises(ValueError, match="mean_sea_level_pressure"):
        continue_through_transition(matched_until(4, centres), fields, INIT_TIME)


def test_a_low_between_grid_points_is_placed_between_them():
    centres = [[(35.4, 280.3 + 2 * step)] for step in range(STEPS)]
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    added = result[result.step > 4]
    assert np.allclose(added.lat, 35.4, atol=0.1)
    assert np.allclose(added.lon, expected_lon(centres, added.step), atol=0.1)


def merging(neighbour_depth: float) -> tuple[pd.DataFrame, xr.Dataset]:
    """Our Gaussian vanishes after step 6 beside a persistent neighbour 3 degrees north."""
    ours = eastward(280, steps=6) + [[] for _ in range(STEPS - 6)]
    neighbour = [[(38.0, 292.0)] for _ in range(STEPS)]
    fields = low_fields(ours, width=1.5)
    beside = low_fields(neighbour, depth=neighbour_depth, width=1.5)
    mslp = "mean_sea_level_pressure"
    fields[mslp] = fields[mslp] + beside[mslp] - 101000.0
    return matched_until(4, ours), fields


def test_a_track_ends_when_its_low_fills_into_a_deeper_neighbour():
    matched, fields = merging(neighbour_depth=3000.0)
    # At step 6 the superposed Gaussians have only the neighbour's distinct minimum.
    assert continue_through_transition(matched, fields, INIT_TIME).step.max() == 5


def test_a_track_continues_through_a_merger_its_own_low_led():
    matched, fields = merging(neighbour_depth=1000.0)
    result = continue_through_transition(matched, fields, INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result[result.step > 6].lat, 38.0, atol=0.1)


def test_a_deeper_neighbour_entering_the_search_radius_is_not_adopted():
    ours = [[(35.0, 280.0)]] * 4 + [[]] * (STEPS - 4)
    neighbour = [[(35.0, 285.0)]] * 4 + [[(35.0, 283.0)]] * (STEPS - 4)
    fields = low_fields(ours, width=1.0)
    beside = low_fields(neighbour, depth=3000.0, width=1.0)
    fields["mean_sea_level_pressure"] += beside.mean_sea_level_pressure - 101000.0
    result = continue_through_transition(matched_until(4, ours), fields, INIT_TIME)
    assert result.step.max() == 4


def test_fractional_longitude_wraps_wind_interpolation_at_the_seam():
    centres = [[(35.0, 359.7)]] * STEPS
    fields = low_fields(centres)
    wind = fields["cyclone_usa_wind_disc"].to_numpy()
    wind[..., 359] = 40.0
    wind[..., 0] = 60.0
    result = continue_through_transition(matched_until(4, centres), fields, INIT_TIME)
    added = result[result.step > 4]
    assert len(added) == STEPS - 4
    expected_knots = 40 + 20 * (added.lon - 359)
    assert np.allclose(added.wind_ms, expected_knots * 1852 / 3600)


def test_subgrid_refinement_cannot_move_a_low_outside_the_search_radius():
    centres = [[(35.0, 280.0)]] * 4 + [[(35.0, 284.45)]] * (STEPS - 4)
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    assert result.step.max() == 4


def test_a_missing_own_minimum_is_not_an_infinitely_shallow_low():
    centres = [[(35.0, 282.3)]] * STEPS
    matched = matched_until(4, [[(35.0, 280.0)]] * STEPS)
    result = continue_through_transition(matched, low_fields(centres), INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result[result.step > 4].lon, 282.3, atol=0.1)


def test_continued_pressure_comes_from_mslp_not_the_tropical_pressure_field():
    centres = [[(35.0, 280.0)]] * STEPS
    fields = low_fields(centres, depth=1000.0)
    result = continue_through_transition(matched_until(4, centres), fields, INIT_TIME)
    added = result[result.step > 4]
    assert len(added) == STEPS - 4
    assert np.allclose(added.mslp_pa, 100000.0)


@pytest.mark.parametrize("longitude", [282.4, 282.0, 281.0])
def test_a_deeper_neighbour_crossing_half_the_gap_is_not_adopted(longitude: float):
    ours = [[(35.0, 280.0)]] * 4 + [[]] * (STEPS - 4)
    neighbour = [[(35.0, 285.0)]] * 4 + [[(35.0, longitude)]] * (STEPS - 4)
    fields = low_fields(ours, width=1.0)
    fields["mean_sea_level_pressure"] += (
        low_fields(neighbour, depth=3000.0, width=1.0).mean_sea_level_pressure - 101000.0
    )
    assert continue_through_transition(matched_until(4, ours), fields, INIT_TIME).step.max() == 4


def test_a_missing_own_low_does_not_allow_an_outside_neighbour_to_take_over():
    ours = [[(35.0, 280.0)]] * 4 + [[]] * (STEPS - 4)
    neighbour = [[(35.0, 285.0)]] * 4 + [[(35.0, 283.0)]] * (STEPS - 4)
    fields = low_fields(ours, depth=100.0, width=1.0)
    fields["mean_sea_level_pressure"] += (
        low_fields(neighbour, depth=3000.0, width=1.0).mean_sea_level_pressure - 101000.0
    )
    assert continue_through_transition(matched_until(4, ours), fields, INIT_TIME).step.max() == 4


def test_a_deeper_neighbour_does_not_stop_a_distinct_moving_own_low():
    ours = [[(35.0, 268.0 + 4 * step)] for step in range(STEPS)]
    neighbour = [[(38.0, 288.0)]] * STEPS
    fields = low_fields(ours, width=1.0)
    fields["mean_sea_level_pressure"] += (
        low_fields(neighbour, depth=3000.0, width=1.0).mean_sea_level_pressure - 101000.0
    )
    result = continue_through_transition(matched_until(4, ours), fields, INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(
        result[result.step > 4].lon, expected_lon(ours, result[result.step > 4].step), atol=0.1
    )


def test_a_grid_minimum_outside_the_radius_can_refine_inside_it():
    centres = [[(35.0, 280.0)]] * 4 + [[(36.6, 283.6)]] * (STEPS - 4)
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result[result.step > 4].lat, 36.6, atol=0.1)
    assert np.allclose(result[result.step > 4].lon, 283.6, atol=0.1)


def test_a_handover_without_a_previous_field_does_not_adopt_a_new_low():
    centres = [[(35.0, 280.0)]] * STEPS
    matched = matched_until(4, centres)
    fields = low_fields(centres).isel(time=slice(4, None))
    assert continue_through_transition(matched, fields, INIT_TIME).step.max() == 4


def test_no_previous_closed_low_does_not_start_a_continuation():
    centres = [[(35.0, 280.0)]] * STEPS
    absent: list[list[tuple[float, float]]] = [[]] * 4
    fields = low_fields(absent + centres[4:])
    assert continue_through_transition(matched_until(4, centres), fields, INIT_TIME).step.max() == 4


def test_a_fast_own_low_can_follow_its_extrapolation_beyond_four_hundred_km_drift():
    centres = [[(35.0, 260.0 + 6 * step)] for step in range(STEPS)]
    result = continue_through_transition(matched_until(4, centres), low_fields(centres), INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result.lon, expected_lon(centres, result.step))


@pytest.mark.parametrize("depth", [1000.0, 3000.0])
def test_a_bystander_does_not_let_a_new_low_start_the_track(depth: float):
    ours = [[(35.0, 280.0)]] * STEPS
    absent: list[list[tuple[float, float]]] = [[]] * 4
    fields = low_fields(absent + ours[4:], width=0.7)
    fields["mean_sea_level_pressure"] += (
        low_fields([[(35.0, 284.0)]] * STEPS, depth=depth, width=0.7).mean_sea_level_pressure
        - 101000.0
    )
    assert continue_through_transition(matched_until(4, ours), fields, INIT_TIME).step.max() == 4


def test_distinct_lows_moving_together_do_not_end_the_own_track():
    ours = [[(35.0, 271.0 + 3 * step)] for step in range(STEPS)]
    neighbour = [[(35.0, 276.0 + 3 * step)] for step in range(STEPS)]
    fields = low_fields(ours, width=1.0)
    fields["mean_sea_level_pressure"] += (
        low_fields(neighbour, depth=3000.0, width=1.0).mean_sea_level_pressure - 101000.0
    )
    result = continue_through_transition(matched_until(4, ours), fields, INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(
        result[result.step > 4].lon, expected_lon(ours, result[result.step > 4].step), atol=0.1
    )


@pytest.mark.parametrize("new_neighbour", [False, True])
def test_a_persistent_own_low_does_not_hand_its_track_to_a_shallower_low(*, new_neighbour: bool):
    ours = [[(35.0, float(lon))] for lon in (271, 274, 277, 280)] + [[(35.0, 280.0)]] * 8
    neighbour: list[list[tuple[float, float]]] = [
        [(35.0, 285.5)] if not new_neighbour or step >= 4 else [] for step in range(STEPS)
    ]
    fields = low_fields(ours, width=0.7)
    fields["mean_sea_level_pressure"] += (
        low_fields(neighbour, depth=600.0, width=0.7).mean_sea_level_pressure - 101000.0
    )
    assert continue_through_transition(matched_until(4, ours), fields, INIT_TIME).step.max() == 4


def test_a_stationary_third_low_does_not_hide_a_deeper_incoming_neighbour():
    ours = [[(35.0, 280.0)]] * 4 + [[]] * (STEPS - 4)
    neighbour = [[(35.0, 285.0)]] * 4 + [[(35.0, 283.0)]] * (STEPS - 4)
    fields = low_fields(ours, width=1.0)
    for centres, depth in [(neighbour, 3000.0), ([[(35.0, 289.0)]] * STEPS, 1500.0)]:
        fields["mean_sea_level_pressure"] += (
            low_fields(centres, depth=depth, width=1.0).mean_sea_level_pressure - 101000.0
        )
    assert continue_through_transition(matched_until(4, ours), fields, INIT_TIME).step.max() == 4


@pytest.mark.parametrize("new_third_low", [False, True])
def test_a_distant_third_low_does_not_steal_the_own_assignment(*, new_third_low: bool):
    ours = [[(35.0, 280.0)]] * STEPS
    neighbour: list[list[tuple[float, float]]] = [[(35.0, 283.5)]] * 4 + [[]] * 8
    third = [[(35.0, 272.0)] if not new_third_low or step >= 4 else [] for step in range(STEPS)]
    fields = low_fields(ours, width=0.7)
    for centres, depth in [(neighbour, 600.0), (third, 1500.0)]:
        fields["mean_sea_level_pressure"] += (
            low_fields(centres, depth=depth, width=0.7).mean_sea_level_pressure - 101000.0
        )
    result = continue_through_transition(matched_until(4, ours), fields, INIT_TIME)
    assert result.step.max() == STEPS
    assert np.allclose(result.lon, 280.0, atol=0.1)
