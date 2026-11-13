from itertools import product

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import rankdata, wilcoxon

from stormfork.export.divergence import (
    Divergence,
    DivergenceStep,
    PressureFields,
    RunName,
    adjust_sweep,
    compute_divergence,
    holm_adjust,
    paired_pvalue,
)
from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.track.match import ensemble_mean


def synthetic_tracks(offset: float = 0, *, members: int = 8) -> pd.DataFrame:
    member, step = np.meshgrid(np.arange(members), np.arange(5), indexing="ij")
    return pd.DataFrame(
        {
            "member": member.ravel(),
            "step": step.ravel(),
            "valid_time": pd.Timestamp("2024-10-07") + pd.to_timedelta(step.ravel() * 6, unit="h"),
            "lat": 20 + member.ravel() / 10,
            "lon": 300 + step.ravel() + offset * (member.ravel() + 1),
            "mslp_pa": 98000 + member.ravel() * 100 + step.ravel() * 10,
            "wind_ms": 40 + member.ravel(),
            "track_id": "target",
        }
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_positive_and_negative_pairs_against_scipy(sign: int):
    differences = sign * np.arange(1, 9, dtype=float)
    expected = wilcoxon(differences, alternative="greater", method="exact").pvalue
    assert paired_pvalue(differences) == expected
    assert expected == (1 / 2 ** len(differences) if sign > 0 else 1)


@pytest.mark.parametrize(
    "differences", [[1, -2, 3, -4, 5, 6, -7, 8], [0, 1, 1, -2, 2, 3, 3, 4], [-2, 2, 3, 2, 3, 4]]
)
def test_midranks_match_exhaustive_sign_permutations(differences: list[float]):
    nonzero = np.array([value for value in differences if value != 0])
    ranks = rankdata(np.abs(nonzero))
    observed = sum(ranks[nonzero > 0])
    totals = [
        sum(ranks[np.array(signs, dtype=bool)])
        for signs in product([False, True], repeat=len(nonzero))
    ]
    assert paired_pvalue(differences) == sum(total >= observed for total in totals) / len(totals)


def test_degenerate_pairs_and_minimum_count():
    assert paired_pvalue(np.zeros(8)) == 1
    assert paired_pvalue(np.ones(4)) is None
    assert (
        paired_pvalue(np.arange(1, 6))
        == wilcoxon(np.arange(1, 6), alternative="greater", method="exact").pvalue
    )


def test_tied_midranks_use_the_actual_sign_permutation_distribution():
    assert paired_pvalue([-2, 2, 3, 2, 3, 4]) == 0.0625


@pytest.mark.parametrize("values", [[np.nan] * 8, [np.inf] * 8, [[1] * 8]])
def test_bad_paired_inputs(values: list[float] | list[list[int]]):
    with pytest.raises(ValueError, match="finite one-dimensional"):
        paired_pvalue(values)


def test_holm_fixed_vector_against_rank_formula():
    pvalues = np.array([0.004, 0.008, 0.02, 0.03, 0.04, 0.2])
    multipliers = len(pvalues) + 1 - rankdata(pvalues, method="ordinal")
    expected = [min(1, max((pvalues * multipliers)[pvalues <= p])) for p in pvalues]
    assert holm_adjust(pvalues.tolist()) == pytest.approx(expected)
    assert holm_adjust([None, *pvalues[::-1], None]) == [None, *reversed(expected), None]
    assert holm_adjust([None, None]) == [None, None]
    assert holm_adjust([0.8, 0.8]) == [1, 1]


@pytest.mark.parametrize("pvalue", [-0.1, 1.1, np.nan, np.inf])
def test_holm_rejects_invalid_probabilities(pvalue: float):
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust([pvalue])


def test_sweep_corrects_each_step_only_over_published_non_null_stops():
    stops = {
        "first": Divergence(
            grid_resolution_deg=1,
            intensity={},
            steps=[
                DivergenceStep(step=0, valid_time=pd.Timestamp("2024-10-07"), paired_p=0.02),
                DivergenceStep(step=1, valid_time=pd.Timestamp("2024-10-07T06:00"), paired_p=None),
            ],
        ),
        "second": Divergence(
            grid_resolution_deg=1,
            intensity={},
            steps=[
                DivergenceStep(step=0, valid_time=pd.Timestamp("2024-10-07"), paired_p=0.03),
                DivergenceStep(step=1, valid_time=pd.Timestamp("2024-10-07T06:00"), paired_p=0.01),
            ],
        ),
    }
    adjusted = adjust_sweep(stops)
    assert [adjusted[name][0].paired_p_holm for name in stops] == holm_adjust([0.02, 0.03])
    assert adjusted["first"][1].paired_p_holm is None
    assert adjusted["second"][1].paired_p_holm == 0.01
    assert stops["first"].steps[0].paired_p_holm is None
    excluded = adjust_sweep(stops, null_stop="first")
    assert excluded["first"][0].paired_p_holm is None
    assert excluded["first"][0].paired_p == stops["first"].steps[0].paired_p
    assert excluded["second"][0].paired_p_holm == stops["second"].steps[0].paired_p
    with pytest.raises(ValueError, match="published stop"):
        adjust_sweep(stops, null_stop="absent")
    stops["second"].steps[0].valid_time += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="valid_time"):
        adjust_sweep(stops)


# A divergence pass over synthetic ensembles outruns the 200 ms default when the machine is loaded.
@settings(deadline=None)
@given(st.floats(min_value=-2, max_value=2, allow_nan=False))
def test_identical_runs_have_zero_divergence(offset: float):
    tracks = synthetic_tracks(offset)
    result = compute_divergence({"baseline": tracks, "fork": tracks, "control": tracks})
    for row in result.steps:
        assert row.fork_sep_km == row.control_sep_km == row.signal_to_noise == 0
        assert row.paired_p == 1
    assert result.intensity_drift_24h_hpa == 0


def test_separation_uses_ensemble_means_but_test_uses_paired_members():
    baseline, fork, control = synthetic_tracks(), synthetic_tracks(1), synthetic_tracks(0.1)
    tracks: dict[RunName, pd.DataFrame] = {"baseline": baseline, "fork": fork, "control": control}
    result = compute_divergence(tracks)
    means = {name: ensemble_mean(frame) for name, frame in tracks.items()}
    for row in result.steps:
        b, f, c = (means[name].iloc[row.step] for name in ("baseline", "fork", "control"))
        expected_sep = float(great_circle_km(b.lat, b.lon, f.lat, f.lon))
        expected_control = float(great_circle_km(b.lat, b.lon, c.lat, c.lon))
        assert row.fork_sep_km == pytest.approx(expected_sep)
        assert row.control_sep_km == pytest.approx(expected_control)
        assert row.signal_to_noise == pytest.approx(
            expected_sep / max(expected_control, np.deg2rad(1) * EARTH_RADIUS_KM)
        )
        members = [
            frame.loc[frame.step == row.step].sort_values("member") for frame in tracks.values()
        ]
        b_members, f_members, c_members = members
        differences = great_circle_km(
            b_members.lat, b_members.lon, f_members.lat, f_members.lon
        ) - great_circle_km(b_members.lat, b_members.lon, c_members.lat, c_members.lon)
        assert row.paired_p == wilcoxon(differences, alternative="greater", method="exact").pvalue
        assert row.baseline_spread_km == pytest.approx(
            float(great_circle_km(b.lat, b.lon, b_members.lat, b_members.lon).mean())
        )


def test_pairing_intersects_member_ids_and_ignores_row_order():
    baseline, fork, control = synthetic_tracks(), synthetic_tracks(0.4), synthetic_tracks(0.2)
    control = control.loc[control.member >= 3].sample(frac=1, random_state=11)
    fork = fork.sample(frac=1, random_state=12)
    full = compute_divergence({"baseline": baseline, "fork": fork, "control": control})
    subset = compute_divergence(
        {
            "baseline": baseline.loc[baseline.member >= 3],
            "fork": fork.loc[fork.member >= 3],
            "control": control,
        }
    )
    assert [row.paired_members for row in full.steps] == [5] * 5
    assert [row.paired_p for row in full.steps] == [row.paired_p for row in subset.steps]
    control = control.loc[control.member >= 4]
    sparse = compute_divergence({"baseline": baseline, "fork": fork, "control": control})
    assert all(row.paired_p is None and row.control_sep_km is None for row in sparse.steps)
    assert all(point.count == 4 for point in sparse.intensity["control"])


def test_intensities_and_24h_drift_use_actual_times_and_member_means():
    baseline = synthetic_tracks()
    fork = baseline.copy()
    fork["mslp_pa"] += 200 + fork.step * 50
    fork["wind_ms"] -= fork.step
    result = compute_divergence({"baseline": baseline, "fork": fork})
    for point in result.intensity["fork"]:
        members = fork.loc[fork.step == point.step]
        assert point.min_mslp_pa == members.mslp_pa.mean()
        assert point.max_wind_ms == members.wind_ms.mean()
    expected = (
        (fork.loc[fork.step == 4].mslp_pa.mean() - baseline.loc[baseline.step == 4].mslp_pa.mean())
        - (
            fork.loc[fork.step == 0].mslp_pa.mean()
            - baseline.loc[baseline.step == 0].mslp_pa.mean()
        )
    ) / 100
    assert result.intensity_drift_24h_hpa == expected
    fork = fork.loc[fork.step != 4]
    assert compute_divergence({"baseline": baseline, "fork": fork}).intensity_drift_24h_hpa is None


def test_pressure_rmse_uses_ensemble_mean_and_1500km_window():
    grid = Grid(np.array([0, 20, 40.0]), np.array([0, 300, 350.0]), 1)
    baseline = np.zeros((5, 8, 3, 3))
    fork = baseline + 1e6
    fork[:, :, 1, 1] = np.arange(8)[None, :] * 10
    tracks = synthetic_tracks()
    result = compute_divergence(
        {"baseline": tracks, "fork": tracks}, mslp=PressureFields(grid, baseline, fork)
    )
    expected = float(np.sqrt(np.mean((np.mean(np.arange(8) * 10)) ** 2)))
    assert all(row.mslp_rmse_pa == expected for row in result.steps)
    with pytest.raises(ValueError, match="supplied grid"):
        compute_divergence(
            {"baseline": tracks}, mslp=PressureFields(grid, baseline, fork[:, :, :-1])
        )
    fork[:, :, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_divergence({"baseline": tracks}, mslp=PressureFields(grid, baseline, fork))


def test_sparse_empty_and_hindcast_tracks_keep_missing_comparisons_null():
    tracks = synthetic_tracks(members=4)
    sparse = compute_divergence({"baseline": tracks, "fork": tracks, "control": tracks})
    assert all(row.fork_sep_km is None and row.baseline_spread_km is None for row in sparse.steps)
    assert len(sparse.intensity["baseline"]) == 5
    empty = compute_divergence({"baseline": tracks.iloc[:0]})
    assert empty.steps == []
    assert empty.intensity == {"baseline": []}
    hindcast = compute_divergence({"baseline": synthetic_tracks()})
    assert all(row.fork_sep_km is None and row.paired_p is None for row in hindcast.steps)


@pytest.mark.parametrize("resolution", [0, -1, np.inf, np.nan])
def test_invalid_grid_floor(resolution: float):
    with pytest.raises(ValueError, match="grid_resolution"):
        compute_divergence({"baseline": synthetic_tracks()}, grid_resolution_deg=resolution)


def test_mismatched_times_and_invalid_intensity_are_rejected():
    baseline, fork = synthetic_tracks(), synthetic_tracks()
    fork["valid_time"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="one-to-one"):
        compute_divergence({"baseline": baseline, "fork": fork})
    baseline.loc[0, "mslp_pa"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_divergence({"baseline": baseline})
