from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stormfork.grid import EARTH_RADIUS_KM, great_circle_km
from stormfork.track import ensemble_mean, match_storm


def synthetic_track(
    track_id: str = "storm", *, member: int = 0, start: int = 0, length: int = 5, lon: float = 300
) -> pd.DataFrame:
    steps = np.arange(start, start + length)
    return pd.DataFrame(
        {
            "member": member,
            "step": steps,
            "valid_time": pd.Timestamp("2024-10-07") + pd.to_timedelta(steps * 6, unit="h"),
            "lat": np.full(length, 20.0),
            "lon": lon + steps.astype(float),
            "mslp_pa": np.full(length, 98000.0),
            "wind_ms": np.full(length, 40.0),
            "track_id": track_id,
        }
    )


@pytest.mark.parametrize("at_init", [False, True])
def test_exact_match(*, at_init: bool):
    predicted = synthetic_track()
    result = match_storm(predicted, predicted, initial_position=(20, 300) if at_init else None)
    pd.testing.assert_frame_equal(result, predicted)


def test_two_tracks_choose_smallest_mean_distance():
    observed = synthetic_track()
    near = synthetic_track("near", lon=301)
    farther = synthetic_track("farther", lon=302)
    result = match_storm(pd.concat([farther, near]), observed)
    pd.testing.assert_frame_equal(result, near)


def test_none_within_500_km():
    observed = synthetic_track()
    assert match_storm(synthetic_track(lon=310), observed).empty


def test_genesis_after_init_uses_common_times():
    observed = synthetic_track(start=2, length=4)
    predicted = synthetic_track(start=1, length=7)
    pd.testing.assert_frame_equal(match_storm(predicted, observed), predicted)


def test_four_common_times_required():
    observed = synthetic_track()
    predicted = synthetic_track(length=3)
    assert match_storm(predicted, observed).empty
    predicted = synthetic_track(length=4)
    pd.testing.assert_frame_equal(match_storm(predicted, observed), predicted)


def test_every_common_time_must_be_within_500_km():
    predicted = synthetic_track()
    observed = predicted.copy(deep=True)
    predicted.loc[4, "lon"] = float(predicted.lon.iloc[4]) + 6
    assert match_storm(predicted, observed).empty


def test_moved_fork_retains_track_1500_km_from_observations():
    observed = synthetic_track()
    moved = synthetic_track("moved")
    moved["lat"] = np.linspace(19, 20 + np.rad2deg(1500 / EARTH_RADIUS_KM), len(moved))
    matched = match_storm(moved, observed, initial_position=(19, 300))
    pd.testing.assert_frame_equal(matched, moved)
    last = matched.iloc[-1]
    assert float(great_circle_km(last.lat, last.lon, 20, last.lon)) == pytest.approx(1500)


def test_initial_matching_ignores_later_observed_path_and_nearby_later_track():
    target = synthetic_track("target")
    target.loc[1:, "lat"] += 15
    other = synthetic_track("other", start=1)
    matched = match_storm(pd.concat([other, target]), pd.DataFrame(), initial_position=(20, 300))
    pd.testing.assert_frame_equal(matched, target)


def test_initial_matching_never_falls_back_to_later_observations():
    observed = synthetic_track()
    predicted = observed.copy(deep=True)
    predicted.loc[0, "lat"] = 30
    assert match_storm(predicted, observed, initial_position=(20, 300)).empty


def test_initial_matching_selects_the_only_track_inside_radius():
    target, other = synthetic_track("target"), synthetic_track("other", lon=310)
    result = match_storm(pd.concat([other, target]), other, initial_position=(20, 300))
    pd.testing.assert_frame_equal(result, target)


@pytest.mark.parametrize(("distance_km", "qualifies"), [(199.99, True), (200.01, False)])
def test_initial_distance_threshold(distance_km: float, *, qualifies: bool):
    predicted = synthetic_track()
    latitude = 20 + np.rad2deg(distance_km / EARTH_RADIUS_KM)
    result = match_storm(predicted, pd.DataFrame(), initial_position=(latitude, 300))
    assert (not result.empty) == qualifies


@pytest.mark.parametrize(("distance_km", "qualifies"), [(499.99, True), (500.01, False)])
def test_genesis_distance_threshold(distance_km: float, *, qualifies: bool):
    observed = synthetic_track()
    predicted = observed.copy(deep=True)
    predicted["lat"] += np.rad2deg(distance_km / EARTH_RADIUS_KM)
    result = match_storm(predicted, observed)
    assert (not result.empty) == qualifies


def test_members_match_independently_with_member_local_track_ids():
    observed = synthetic_track()
    predicted = pd.concat(
        [synthetic_track(member=0), synthetic_track(member=1, lon=310), synthetic_track(member=2)]
    )
    matched = match_storm(predicted, observed)
    assert matched.member.unique().tolist() == [0, 2]
    assert len(matched) == 10


def test_longitude_wrap_and_great_circle_distance():
    observed = synthetic_track(lon=-60)
    predicted = synthetic_track()
    pd.testing.assert_frame_equal(match_storm(predicted, observed), predicted)
    predicted["lat"], observed["lat"] = 80.0, 80.0
    predicted["lon"] += 10
    assert len(match_storm(predicted, observed)) == len(predicted)


@pytest.mark.parametrize("at_init", [False, True])
def test_ambiguous_identity_raises(*, at_init: bool):
    first, second = synthetic_track("first"), synthetic_track("second")
    with pytest.raises(ValueError, match="multiple tracks"):
        match_storm(
            pd.concat([first, second]), first, initial_position=(20, 300) if at_init else None
        )


def test_ensemble_averages_only_matched_members_and_stores_step_counts():
    observed = synthetic_track()
    tracks = pd.concat([synthetic_track(member=i, length=5 - i // 2) for i in range(8)])
    tracks.loc[tracks.member == 7, "lon"] += 20
    tracks["mslp_pa"] += tracks.member * 100
    tracks["wind_ms"] += tracks.member
    mean = ensemble_mean(match_storm(tracks, observed, initial_position=(20, 300)))
    assert mean.step.tolist() == [0, 1, 2]
    assert mean["count"].tolist() == [7, 7, 6]
    assert mean.lat.tolist() == [20, 20, 20]
    assert mean.lon.tolist() == pytest.approx([300, 301, 302])
    assert mean.mslp_pa.tolist() == pytest.approx([98300, 98300, 98250])
    assert mean.wind_ms.tolist() == pytest.approx([43, 43, 42.5])


@pytest.mark.parametrize("members", [4, 5, 8])
def test_five_of_eight_rule(members: int):
    tracks = pd.concat([synthetic_track(member=i) for i in range(members)])
    mean = ensemble_mean(tracks)
    assert len(mean) == (5 if members >= 5 else 0)
    if members >= 5:
        assert mean["count"].tolist() == [members] * 5


def test_ensemble_longitudes_cross_seam():
    tracks = pd.concat([synthetic_track(member=i, lon=359 if i < 3 else 1) for i in range(6)])
    mean = ensemble_mean(tracks)
    assert ((mean.lon >= 0) & (mean.lon < 360)).all()
    assert np.allclose(np.minimum(mean.lon, 360 - mean.lon), [0, 1, 2, 3, 4])


def test_ensemble_does_not_fill_a_missing_step():
    tracks = pd.concat([synthetic_track(member=i) for i in range(8)])
    tracks = tracks.loc[(tracks.step != 2) | (tracks.member < 4)]
    mean = ensemble_mean(tracks)
    assert mean.step.tolist() == [0, 1, 3, 4]
    assert mean["count"].tolist() == [8, 8, 8, 8]


def test_empty_tracks_preserve_output_columns():
    empty = synthetic_track().iloc[:0]
    matched = match_storm(empty, empty)
    pd.testing.assert_frame_equal(matched, empty)
    mean = ensemble_mean(matched)
    assert mean.empty
    assert mean.columns.tolist() == [
        "step",
        "valid_time",
        "lat",
        "lon",
        "mslp_pa",
        "wind_ms",
        "count",
    ]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("duplicate", "one position"),
        ("missing", "missing columns"),
        ("nan", "finite"),
        ("step", "nonnegative integers"),
        ("time", "non-missing datetimes"),
    ],
)
def test_invalid_track_tables(operation: str, message: str):
    predicted = synthetic_track()
    if operation == "duplicate":
        predicted = pd.concat([predicted, predicted.iloc[:1]])
    elif operation == "missing":
        predicted = predicted.drop(columns="track_id")
    elif operation == "nan":
        predicted.loc[0, "lat"] = np.nan
    elif operation == "step":
        predicted["step"] = predicted.step.astype(float)
        predicted.loc[0, "step"] = 0.5
    else:
        predicted.loc[0, "valid_time"] = pd.NaT
    with pytest.raises(ValueError, match=message):
        match_storm(predicted, synthetic_track())
    with pytest.raises(ValueError, match=message):
        ensemble_mean(predicted)


def test_duplicate_observations_cannot_count_as_four_common_times():
    observed = pd.concat([synthetic_track(length=2)] * 2)
    with pytest.raises(ValueError, match="one position per valid_time"):
        match_storm(synthetic_track(), observed)


def test_ensemble_rejects_multiple_tracks_for_a_member():
    tracks = pd.concat([synthetic_track(), synthetic_track("other", start=5)])
    with pytest.raises(ValueError, match="one matched track per member"):
        ensemble_mean(tracks)


def test_conflicting_step_times_rejected():
    tracks = pd.concat([synthetic_track(member=0), synthetic_track(member=1)])
    tracks.loc[tracks.member == 1, "valid_time"] += pd.Timedelta(hours=6)
    with pytest.raises(ValueError, match="one-to-one"):
        ensemble_mean(tracks)


def test_track_restarted_after_one_missing_fix_continues():
    first = synthetic_track("first", length=3)
    restarted = synthetic_track("restarted", start=4, length=4)
    matched = match_storm(pd.concat([restarted, first]), pd.DataFrame(), initial_position=(20, 300))
    assert matched.step.tolist() == [0, 1, 2, 4, 5, 6, 7]
    assert matched.track_id.unique().tolist() == ["first"]


def test_continuation_chains_across_several_gaps():
    parts = [synthetic_track(str(i), start=start, length=2) for i, start in enumerate([0, 3, 6])]
    matched = match_storm(pd.concat(parts), pd.DataFrame(), initial_position=(20, 300))
    assert matched.step.tolist() == [0, 1, 3, 4, 6, 7]


def test_two_missing_fixes_end_the_track():
    first = synthetic_track("first", length=3)
    restarted = synthetic_track("restarted", start=5, length=3)
    matched = match_storm(pd.concat([first, restarted]), pd.DataFrame(), initial_position=(20, 300))
    pd.testing.assert_frame_equal(matched, first)


@pytest.mark.parametrize(("offset_km", "joined"), [(199.0, True), (201.0, False)])
def test_continuation_distance_is_measured_from_the_extrapolated_position(
    offset_km: float, *, joined: bool
):
    first = synthetic_track("first", length=3)
    restarted = synthetic_track("restarted", start=4, length=3)
    restarted["lat"] += np.rad2deg(offset_km / EARTH_RADIUS_KM)
    matched = match_storm(pd.concat([first, restarted]), pd.DataFrame(), initial_position=(20, 300))
    assert (len(matched) > len(first)) == joined


def test_continuation_picks_the_candidate_nearest_the_extrapolation():
    first = synthetic_track("first", length=3)
    near = synthetic_track("near", start=4, length=3)
    far = synthetic_track("far", start=4, length=3)
    far["lat"] += 1.0
    matched = match_storm(pd.concat([far, first, near]), pd.DataFrame(), initial_position=(20, 300))
    assert matched.step.tolist() == [0, 1, 2, 4, 5, 6]
    assert matched.lat.tolist() == [20.0] * 6


def test_continuation_crosses_the_longitude_seam():
    first = synthetic_track("first", length=3, lon=358)
    restarted = synthetic_track("restarted", start=4, length=3, lon=358)
    restarted["lon"] %= 360
    matched = match_storm(pd.concat([first, restarted]), pd.DataFrame(), initial_position=(20, 358))
    assert matched.step.tolist() == [0, 1, 2, 4, 5, 6]


def test_sandy_baseline_member_keeps_the_storm_across_a_restart():
    raw = pd.read_csv(
        Path(__file__).parent / "fixtures/sandy_baseline_member3_raw_tracks.csv",
        parse_dates=["valid_time"],
        dtype={"track_id": str},
    )
    matched = match_storm(raw, pd.DataFrame(), initial_position=(24.8, 284.1))
    assert matched.step.tolist() == [*range(5), *range(6, 18)]
    assert matched.track_id.unique().tolist() == ["2012296N14283"]
