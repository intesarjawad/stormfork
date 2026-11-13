from importlib import import_module
from typing import Never

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stormfork.track import ensemble_mean, match_storm, storm_start, track

INIT_TIME = np.datetime64("2024-10-07T00:00:00", "ns")


def gaussian_fields(*, steps: int = 12, longitude: float = 180) -> xr.Dataset:
    lat, lon = np.arange(-10, 11, dtype=float), np.arange(360, dtype=float)
    longitude_distance = (lon - longitude + 180) % 360 - 180
    gaussian = np.exp(-(lat[:, None] ** 2 + longitude_distance**2) / 2)
    probability = np.broadcast_to(gaussian, (steps, *gaussian.shape)).copy()
    fields = xr.Dataset(
        {
            "cyclone_exists_gaussian_unit_mode": (("time", "lat", "lon"), probability),
            "cyclone_usa_pres_disc": (("time", "lat", "lon"), np.full_like(probability, 980.0)),
            "cyclone_usa_wind_disc": (("time", "lat", "lon"), np.full_like(probability, 60.0)),
            "mean_sea_level_pressure": (("time", "lat", "lon"), 101000 - 3000 * probability),
        },
        coords={"time": np.arange(steps) * np.timedelta64(6, "h"), "lat": lat, "lon": lon},
    )
    fields["cyclone_usa_rmw_disc"] = fields.cyclone_usa_wind_disc * 0 + 20
    for speed in (34, 50, 64):
        for quadrant in ("ne", "se", "sw", "nw"):
            fields[f"cyclone_usa_r{speed}_{quadrant}_radius_disc"] = (
                fields.cyclone_usa_wind_disc * 0 + 100
            )
    return fields


def ibtracs_dataset(
    *, start: int = 0, longitude: float = 180, include_other: bool = False, points: int = 2
) -> xr.Dataset:
    times = INIT_TIME + np.arange(start - 1, start - 1 + points) * np.timedelta64(6, "h")
    ids = ["target", "other"] if include_other else ["target"]
    longitudes = [longitude, longitude + 20] if include_other else [longitude]
    shape = (len(ids), points)
    return xr.Dataset(
        {
            "time": (("storm", "date_time"), np.broadcast_to(times, shape)),
            "sid": ("storm", ids),
            "lat": (("storm", "date_time"), np.zeros(shape)),
            "lon": (
                ("storm", "date_time"),
                np.repeat(np.array(longitudes)[:, None], points, axis=1),
            ),
            "usa_pres": (("storm", "date_time"), np.full(shape, 980.0)),
            "usa_wind": (("storm", "date_time"), np.full(shape, 60.0)),
            "usa_rmw": (("storm", "date_time"), np.full(shape, 20.0)),
        }
    )


def test_gaussian_low_yields_one_track_and_converts_units():
    fields, observations = gaussian_fields(), ibtracs_dataset()
    before_fields, before_observations = fields.copy(deep=True), observations.copy(deep=True)
    result = track(fields, observations, INIT_TIME)
    assert result.columns.tolist() == [
        "member",
        "step",
        "valid_time",
        "lat",
        "lon",
        "mslp_pa",
        "wind_ms",
        "track_id",
    ]
    assert result.track_id.unique().tolist() == ["target"]
    assert result.member.unique().tolist() == [0]
    assert result.step.tolist() == list(range(12))
    assert result.valid_time.tolist() == list(pd.date_range(INIT_TIME, periods=12, freq="6h"))
    assert np.allclose(result.lat, 0, atol=0.01)
    assert np.allclose(result.lon, 180, atol=0.01)
    assert np.allclose(result.mslp_pa, 98000)
    assert np.allclose(result.wind_ms, 60 * 1852 / 3600)
    assert fields.identical(before_fields)
    assert observations.identical(before_observations)


def test_genesis_without_initial_storms():
    result = track(gaussian_fields(steps=13), ibtracs_dataset(start=3), INIT_TIME)
    assert len(result.track_id.unique()) == 1
    assert result.step.tolist() == list(range(1, 13))
    assert np.allclose(result.lon, 180, atol=0.01)


def test_a_target_that_forms_after_the_init_is_matched_to_its_observed_track():
    observations = ibtracs_dataset(start=3, points=5)
    fields = gaussian_fields(steps=13).expand_dims(member=range(2))
    result = track(fields, observations, INIT_TIME, target_track_id="target")
    assert result.member.unique().tolist() == [0, 1]
    assert result.groupby("member").track_id.nunique().tolist() == [1, 1]
    assert result.loc[result.member == 0].step.tolist() == list(range(1, 13))


def test_a_target_that_forms_after_the_init_far_from_every_track_is_lost():
    # Ten degrees of longitude on the equator is about 1,100 km, beyond the 500 km gate.
    observations = ibtracs_dataset(start=3, points=5, longitude=190)
    result = track(gaussian_fields(steps=13), observations, INIT_TIME, target_track_id="target")
    assert result.empty


def test_a_target_that_forms_after_the_init_needs_four_common_times():
    observations = ibtracs_dataset(start=3, points=3)
    result = track(gaussian_fields(steps=13), observations, INIT_TIME, target_track_id="target")
    assert result.empty


def test_a_target_absent_from_ibtracs_is_an_error():
    with pytest.raises(ValueError, match="initial position for storm absent"):
        track(gaussian_fields(steps=3), ibtracs_dataset(), INIT_TIME, target_track_id="absent")


def test_short_genesis_track_uses_default_duration_filter():
    result = track(gaussian_fields(steps=4), ibtracs_dataset(start=3), INIT_TIME)
    assert result.empty
    assert result.columns.tolist() == [
        "member",
        "step",
        "valid_time",
        "lat",
        "lon",
        "mslp_pa",
        "wind_ms",
        "track_id",
    ]
    assert ensemble_mean(match_storm(result, result)).empty


def test_no_storm_forms_in_zero_probability_fields():
    fields = gaussian_fields(steps=3)
    fields["cyclone_exists_gaussian_unit_mode"] *= 0
    result = track(fields, ibtracs_dataset(start=3), INIT_TIME)
    assert result.empty
    assert ensemble_mean(match_storm(result, result)).empty


def test_other_storms_at_the_init_are_tracked_without_a_target():
    result = track(gaussian_fields(steps=3), ibtracs_dataset(include_other=True), INIT_TIME)
    other_initial = result.loc[(result.track_id == "other") & (result.step == 0)]
    assert other_initial[["lat", "lon", "mslp_pa"]].to_numpy().tolist() == [[0, 200, 98000]]


def test_a_target_keeps_only_its_matched_track_so_members_average():
    fields = gaussian_fields(steps=3).expand_dims(member=range(5))
    result = track(fields, ibtracs_dataset(include_other=True), INIT_TIME, target_track_id="target")
    assert result.track_id.unique().tolist() == ["target"]
    assert ensemble_mean(result)["count"].tolist() == [5, 5, 5]


def test_relocation_keeps_the_moved_target_and_drops_the_unmoved_storm():
    observations = ibtracs_dataset(longitude=1, include_other=True)
    before = observations.copy(deep=True)
    result = track(
        gaussian_fields(steps=3, longitude=359),
        observations,
        INIT_TIME,
        target_track_id="target",
        moved_initial_position=(0, 359),
    )
    assert result.track_id.unique().tolist() == ["target"]
    assert result.step.tolist() == [0, 1, 2]
    assert np.allclose(result.lon, 359, atol=0.01)
    assert observations.identical(before)


def test_relocation_changes_target_initialization_and_wraps_longitude():
    result = track(
        gaussian_fields(steps=3, longitude=359),
        ibtracs_dataset(longitude=1),
        INIT_TIME,
        target_track_id="target",
        moved_initial_position=(0, -1),
    )
    assert result.track_id.unique().tolist() == ["target"]
    assert np.allclose(result.lon, 359, atol=0.01)
    assert result.step.tolist() == [0, 1, 2]


def test_storm_start_reads_the_storm_at_the_init_and_its_first_record():
    observations = ibtracs_dataset(longitude=1, include_other=True)
    start = storm_start(observations, INIT_TIME, "other")
    assert start.position == (0, 21)
    assert start.first_record_time == pd.Timestamp(INIT_TIME - np.timedelta64(6, "h"))
    with pytest.raises(ValueError, match="no record of storm absent"):
        storm_start(observations, INIT_TIME, "absent")


def test_storm_start_reads_the_gale_radii_in_nautical_miles():
    observations = ibtracs_dataset()
    quadrants = np.broadcast_to([200.0, np.nan, 150.0, 100.0], (1, 2, 4))
    observations["usa_r34"] = (("storm", "date_time", "quadrant"), quadrants)
    radii = storm_start(observations, INIT_TIME, "target").gale_radii_nm
    np.testing.assert_allclose(radii, [200, np.nan, 150, 100])
    assert storm_start(ibtracs_dataset(), INIT_TIME, "target").gale_radii_nm == pytest.approx(
        [np.nan] * 4, nan_ok=True
    )


def test_storm_start_before_genesis_has_no_position():
    start = storm_start(ibtracs_dataset(start=3), INIT_TIME, "target")
    assert start.position is None
    assert start.gale_radii_nm == ()
    assert start.first_record_time == pd.Timestamp(INIT_TIME + np.timedelta64(12, "h"))


def test_storm_start_after_the_last_record_is_an_error():
    with pytest.raises(ValueError, match="no single initial position for storm target"):
        storm_start(ibtracs_dataset(start=-3), INIT_TIME, "target")


def test_storm_start_tolerates_times_decoded_off_the_hour():
    observations = ibtracs_dataset(include_other=True)
    observations["time"] = observations["time"] + np.timedelta64(39936, "ns")
    assert storm_start(observations, INIT_TIME, "other").position == (0, 200)


def test_multiple_members_and_singleton_batch():
    fields = gaussian_fields(steps=2).expand_dims(member=[2, 7], batch=[0])
    result = track(fields, ibtracs_dataset(), INIT_TIME)
    assert result.member.tolist() == [2, 2, 7, 7]
    assert result.step.tolist() == [0, 1, 0, 1]


def test_forecast_only_fields_include_initial_storm():
    fields = gaussian_fields(steps=3).isel(time=slice(1, None))
    result = track(fields, ibtracs_dataset(), INIT_TIME)
    assert result.step.tolist() == [0, 1, 2]


def test_missing_backend_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    module = import_module("stormfork.track.tracker")

    def unavailable(name: str) -> Never:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"WeatherNext.*uv sync --group model"):
        track(gaussian_fields(), ibtracs_dataset(), INIT_TIME)


@pytest.mark.parametrize("leads", [[0], [0, 12], [-6, 0, 6], [12, 18]])
def test_invalid_lead_times(leads: list[int]):
    fields = gaussian_fields(steps=len(leads))
    fields["time"] = np.array(leads) * np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="lead times"):
        track(fields, ibtracs_dataset(), INIT_TIME)


def test_relocation_requires_target_id_and_present_storm():
    with pytest.raises(ValueError, match="target_track_id"):
        track(gaussian_fields(), ibtracs_dataset(), INIT_TIME, moved_initial_position=(0, 179))
    with pytest.raises(ValueError, match="initial position for storm absent"):
        track(
            gaussian_fields(),
            ibtracs_dataset(),
            INIT_TIME,
            target_track_id="absent",
            moved_initial_position=(0, 179),
        )


@pytest.mark.parametrize("members", [[0, 0], [-1, 0], [0, 0.5]])
def test_invalid_member_labels(members: list[float]):
    with pytest.raises(ValueError, match="member labels"):
        track(gaussian_fields().expand_dims(member=members), ibtracs_dataset(), INIT_TIME)


def test_missing_scalar_field_names_the_output():
    fields = gaussian_fields().drop_vars("cyclone_usa_rmw_disc")
    with pytest.raises(ValueError, match="missing cyclone outputs: cyclone_usa_rmw_disc"):
        track(fields, ibtracs_dataset(), INIT_TIME)


def test_time_coordinate_requires_timedeltas():
    fields = gaussian_fields(steps=2)
    fields["time"] = [0, 6]
    with pytest.raises(ValueError, match="lead timedeltas"):
        track(fields, ibtracs_dataset(), INIT_TIME)


def test_multiple_initializations_are_rejected():
    with pytest.raises(ValueError, match="batch size 1"):
        track(gaussian_fields().expand_dims(batch=[0, 1]), ibtracs_dataset(), INIT_TIME)
