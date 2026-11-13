"""Publish a sweep's stops: each stop's paired statistics, tracks and pressure atlas.

The null value is the sweep's own fork, so its stop reads the experiment's
tracks, divergence and fork atlas. Every other published stop gets a
tracks.json (the baseline, the stop as the fork, and the control), a
divergence.json and its atlas in its run directory. Each stop's paired p is
Holm-adjusted across the published non-null stops at every step.
"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import pandas as pd

from stormfork.export.divergence import Divergence, RunName, adjust_sweep, compute_divergence
from stormfork.export.manifest import PublishedStop, SweepExport, SweepInfo, track_exports

Direction = Literal["east", "west", "north", "south", "none"]
# Positive and negative directions of each displacement parameter.
DIRECTIONS: dict[str, tuple[Direction, Direction]] = {
    "east_km": ("east", "west"),
    "north_km": ("north", "south"),
}
NULL_ATLAS = "fork/mslp.png"
STOP_ATLAS = "mslp.png"


class StopRun(NamedTuple):
    """One published non-null stop: its run directory, value, tracks and atlas bytes."""

    run: str
    value: float
    tracks: pd.DataFrame
    atlas: bytes


def stop_place(param: str, value: float) -> tuple[Direction, float]:
    """Name where a displacement value moves the storm: a direction and a distance in km."""
    positive, negative = DIRECTIONS[param]
    if value == 0:
        return "none", 0.0
    return (positive if value > 0 else negative), abs(value)


def stop_name(param: str, value: float) -> str:
    """Name a stop in divergence.json and the slider, such as east_300, or east_km_1.5."""
    if param not in DIRECTIONS:
        return f"{param}_{value:g}"
    direction, distance = stop_place(param, value)
    return f"{direction}_{distance:g}"


def sweep_evidence(
    param: str,
    null_value: float,
    tracks: Mapping[RunName, pd.DataFrame],
    divergence: Divergence,
    stops: Sequence[StopRun],
) -> tuple[Divergence, SweepExport | None]:
    """Add each stop's paired statistics to the null fork's divergence, and the slider block.

    divergence is the null fork's, computed from tracks. Only a displacement
    sweep gets a slider block; any other sweep is published as statistics.

    Raises:
        ValueError: The tracks lack the baseline or control, or stops disagree about time.
    """
    null = stop_name(param, null_value)
    compared: dict[str, dict[RunName, pd.DataFrame]] = {
        stop.run: {
            "baseline": tracks["baseline"],
            "fork": stop.tracks,
            "control": tracks["control"],
        }
        for stop in stops
    }
    reports = {null: divergence}
    reports.update(
        {stop_name(param, stop.value): compute_divergence(compared[stop.run]) for stop in stops}
    )
    adjusted = adjust_sweep(reports, null_stop=null)
    evidence = divergence.model_copy(update={"steps": adjusted[null], "stops": adjusted})
    if param not in DIRECTIONS:
        return evidence, None
    direction, distance = stop_place(param, null_value)
    published = [
        (
            null_value,
            PublishedStop(
                id=null,
                direction=direction,
                distance_km=distance,
                tracks="tracks.json",
                atlas=NULL_ATLAS,
                divergence="divergence.json",
                sha256={},
            ),
        )
    ]
    files: dict[str, bytes] = {}
    for stop in stops:
        name = stop_name(param, stop.value)
        report = reports[name].model_copy(update={"steps": adjusted[name]})
        own = {
            f"{stop.run}/tracks.json": track_exports(compared[stop.run])["tracks.json"],
            f"{stop.run}/divergence.json": (report.model_dump_json(indent=2) + "\n").encode(),
            f"{stop.run}/{STOP_ATLAS}": stop.atlas,
        }
        files.update(own)
        direction, distance = stop_place(param, stop.value)
        published.append(
            (
                stop.value,
                PublishedStop(
                    id=name,
                    direction=direction,
                    distance_km=distance,
                    tracks=f"{stop.run}/tracks.json",
                    atlas=f"{stop.run}/{STOP_ATLAS}",
                    divergence=f"{stop.run}/divergence.json",
                    sha256={path: hashlib.sha256(data).hexdigest() for path, data in own.items()},
                ),
            )
        )
    published.sort(key=lambda item: item[0])
    info = SweepInfo(name=param, baseline=null, stops=[stop for _, stop in published])
    return evidence, SweepExport(info, files)
