"""Compare a rerun's tracks with the published ones and record the outcome.

A reproduction entry is what a contributor appends to an experiment's
reproductions.json. The tolerance limits are chosen policy, set out in
docs/reproducibility.md; they are not measured spreads.
"""

from collections.abc import Mapping
from datetime import date
from typing import get_args

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from stormfork.export.divergence import EvidenceModel, Nonnegative, RunName
from stormfork.export.manifest import GithubLogin, Text
from stormfork.grid import great_circle_km
from stormfork.track.match import ensemble_mean

SAME_RUNTIME_TOLERANCE_KM = 10.0
CROSS_RUNTIME_TOLERANCE_KM = 50.0
NULL_FORK_TOLERANCE_KM = SAME_RUNTIME_TOLERANCE_KM
SAMPLE_KEY = ["member", "step"]


class Reproduction(EvidenceModel):
    """One contributor's rerun of a published experiment.

    runtime names the compute backend and device of the rerun, and
    max_track_diff_km is the largest member or ensemble-mean position difference.
    """

    by: GithubLogin
    runtime: Text
    date: date
    code_version: Text
    max_track_diff_km: Nonnegative
    passed: bool


Reproductions = TypeAdapter(list[Reproduction])


def tolerance_km(published_kind: str, rerun_kind: str) -> float:
    """Allow more drift between different kinds of hardware than on the same kind."""
    return SAME_RUNTIME_TOLERANCE_KM if published_kind == rerun_kind else CROSS_RUNTIME_TOLERANCE_KM


def _max_difference_km(
    published: pd.DataFrame, rerun: pd.DataFrame, keys: list[str], what: str
) -> float:
    joined = published.merge(rerun, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    if (joined["_merge"] != "both").any():
        msg = f"the rerun and the published {what} do not cover the same samples"
        raise ValueError(msg)
    distances = great_circle_km(joined.lat_a, joined.lon_a, joined.lat_b, joined.lon_b)
    return float(np.max(distances, initial=0.0))


def max_track_difference_km(
    published: Mapping[RunName, pd.DataFrame], rerun: Mapping[RunName, pd.DataFrame]
) -> float:
    """Largest great-circle distance between matching member and ensemble-mean positions.

    Members pair on member and step, ensemble means on step. Nothing is
    interpolated, so a missing or extra sample is an error, not a difference.

    Raises:
        ValueError: The two sides hold different runs, members or steps.
    """
    if set(published) != set(rerun):
        msg = "the rerun and the published experiment hold different runs"
        raise ValueError(msg)
    largest = 0.0
    for run in get_args(RunName):
        if run not in published:
            continue
        members = _max_difference_km(published[run], rerun[run], SAMPLE_KEY, f"{run} member tracks")
        means = _max_difference_km(
            ensemble_mean(published[run]),
            ensemble_mean(rerun[run]),
            ["step"],
            f"{run} ensemble means",
        )
        largest = max(largest, members, means)
    return largest


def null_fork_max_track_diff_km(tracks: Mapping[RunName, pd.DataFrame]) -> float:
    """Largest member or ensemble-mean distance between the fork and the baseline.

    A fork that applies no edit shares the baseline's seeds, so it must land on
    the baseline's tracks up to numerical noise; this is that distance.

    Raises:
        ValueError: Either run is absent or the two cover different samples.
    """
    if "baseline" not in tracks or "fork" not in tracks:
        msg = "the null-fork check needs both the baseline and the fork tracks"
        raise ValueError(msg)
    return max_track_difference_km({"baseline": tracks["baseline"]}, {"baseline": tracks["fork"]})
