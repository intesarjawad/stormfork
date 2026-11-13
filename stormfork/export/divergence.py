from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata

from stormfork.grid import EARTH_RADIUS_KM, Grid, great_circle_km
from stormfork.track.match import MIN_MEAN_MEMBERS, ensemble_mean

RunName = Literal["baseline", "fork", "control"]
Nonnegative = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
PRESSURE_RADIUS_KM = 1500.0
PA_PER_HPA = 100.0
MAX_GRID_SPACING_DEG = 180.0
MEMBER_FIELD_NDIM = 4


class EvidenceModel(BaseModel):
    """Reject unknown keys and nonfinite numbers in published evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DivergenceStep(EvidenceModel):
    """Distances in km and pressure RMSE in Pa; missing comparisons stay null."""

    step: Annotated[int, Field(ge=0)]
    valid_time: datetime
    fork_sep_km: Nonnegative | None = None
    control_sep_km: Nonnegative | None = None
    signal_to_noise: Nonnegative | None = None
    baseline_spread_km: Nonnegative | None = None
    mslp_rmse_pa: Nonnegative | None = None
    paired_members: Annotated[int, Field(ge=0)] = 0
    paired_p: Probability | None = None
    paired_p_holm: Probability | None = None


class IntensityPoint(EvidenceModel):
    """Mean matched-member central pressure (Pa) and peak wind (m/s)."""

    step: Annotated[int, Field(ge=0)]
    valid_time: datetime
    count: Annotated[int, Field(gt=0)]
    min_mslp_pa: Nonnegative
    max_wind_ms: Nonnegative


class Divergence(EvidenceModel):
    """Track-derived evidence, optionally augmented by pressure fields and stops."""

    grid_resolution_deg: Annotated[float, Field(gt=0, le=180)]
    steps: list[DivergenceStep]
    intensity: dict[RunName, list[IntensityPoint]]
    intensity_drift_24h_hpa: float | None = None
    stops: dict[str, list[DivergenceStep]] = Field(default_factory=dict)


def paired_pvalue(differences: ArrayLike) -> float | None:
    """Test positive signed-rank sums against all sign permutations of midranks.

    Fewer than five pairs yield null. An all-zero difference has no evidence
    against the null and yields 1. Zero differences are discarded before ranking.
    Integer doubled midranks preserve ties exactly; subset counts give the full
    sign distribution without enumerating its exponentially many assignments.

    Raises:
        ValueError: Differences are not a finite one-dimensional array.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        msg = "paired differences must be a finite one-dimensional array"
        raise ValueError(msg)
    if values.size < MIN_MEAN_MEMBERS:
        return None
    nonzero = values[values != 0]
    ranks = (2 * rankdata(np.abs(nonzero))).astype(np.int64)
    observed = int(ranks[nonzero > 0].sum())
    counts = {0: 1}
    for rank in ranks:
        for total, count in list(counts.items()):
            shifted = total + int(rank)
            counts[shifted] = counts.get(shifted, 0) + count
    return sum(count for total, count in counts.items() if total >= observed) / 2 ** len(nonzero)


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Apply Holm step-down correction to non-null tests, preserving input order.

    Raises:
        ValueError: A supplied p-value is nonfinite or outside [0, 1].
    """
    present = [(index, p) for index, p in enumerate(pvalues) if p is not None]
    if any(not np.isfinite(p) or not 0 <= p <= 1 for _, p in present):
        msg = "p-values must be finite and in [0, 1]"
        raise ValueError(msg)
    adjusted: list[float | None] = [None] * len(pvalues)
    previous = 0.0
    for rank, (index, pvalue) in enumerate(sorted(present, key=lambda item: item[1])):
        previous = min(1.0, max(previous, (len(present) - rank) * pvalue))
        adjusted[index] = previous
    return adjusted


def adjust_sweep(
    stops: Mapping[str, Divergence], *, null_stop: str | None = None
) -> dict[str, list[DivergenceStep]]:
    """Adjust published non-null edits independently at each step.

    Supply the zero-edit stop's name as null_stop to retain its raw test while
    excluding it from the family. Missing p-values also contribute no test.

    Raises:
        ValueError: A stop is unknown, disagrees about time or repeats a step.
    """
    if null_stop is not None and null_stop not in stops:
        msg = "null_stop must name a published stop"
        raise ValueError(msg)
    result = {name: [row.model_copy() for row in report.steps] for name, report in stops.items()}
    by_step: dict[int, list[DivergenceStep]] = {}
    for name, rows in result.items():
        if len({row.step for row in rows}) != len(rows):
            msg = "sweep stops must not repeat a step"
            raise ValueError(msg)
        for row in rows:
            row.paired_p_holm = None
            if name == null_stop:
                continue
            by_step.setdefault(row.step, []).append(row)
    for rows in by_step.values():
        if len({row.valid_time for row in rows}) != 1:
            msg = "sweep stops disagree about valid_time"
            raise ValueError(msg)
        for row, adjusted in zip(rows, holm_adjust([row.paired_p for row in rows]), strict=True):
            row.paired_p_holm = adjusted
    return result


@dataclass(frozen=True)
class PressureFields:
    """MSLP in Pa on one grid, laid out (step, lat, lon) or (step, member, lat, lon).

    The step axis starts at analysis step 0. I average member fields before
    computing RMSE, so ensemble variability is not counted as an edit signal.
    """

    grid: Grid
    baseline: ArrayLike
    fork: ArrayLike

    def rmse_pa(self, step: int, latitude: float, longitude: float) -> float:
        """Compare equal-weight grid cells within 1500 km of the baseline mean.

        Raises:
            ValueError: Arrays have invalid shapes or the window lacks finite cells.
        """
        fields: list[NDArray[np.float64]] = [
            np.asarray(field, dtype=np.float64) for field in (self.baseline, self.fork)
        ]
        for field in fields:
            if (
                field.ndim not in (3, 4)
                or field.shape[0] <= step
                or field.shape[-2:] != (len(self.grid.lat), len(self.grid.lon))
            ):
                msg = "pressure fields require (step, [member,] lat, lon) on the supplied grid"
                raise ValueError(msg)
        baseline, fork = [
            field[step].mean(axis=0) if field.ndim == MEMBER_FIELD_NDIM else field[step]
            for field in fields
        ]
        distances = great_circle_km(
            latitude, longitude, self.grid.lat[:, None], self.grid.lon[None, :]
        )
        values = np.asarray((fork - baseline)[distances <= PRESSURE_RADIUS_KM], dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            msg = "pressure window must contain finite grid cells within 1500 km"
            raise ValueError(msg)
        return float(np.sqrt(np.mean(values**2)))


def _paired_distances(positions: Mapping[RunName, pd.DataFrame]) -> tuple[int, float | None]:
    if any(name not in positions for name in ("baseline", "fork", "control")):
        return 0, None
    baseline = positions["baseline"].set_index("member")
    fork = positions["fork"].set_index("member")
    control = positions["control"].set_index("member")
    common = baseline.index.intersection(fork.index).intersection(control.index)
    baseline, fork, control = baseline.loc[common], fork.loc[common], control.loc[common]
    differences = great_circle_km(baseline.lat, baseline.lon, fork.lat, fork.lon) - great_circle_km(
        baseline.lat, baseline.lon, control.lat, control.lon
    )
    return len(common), paired_pvalue(differences)


def _drift(intensity: Mapping[RunName, list[IntensityPoint]]) -> float | None:
    baseline = {point.valid_time: point for point in intensity.get("baseline", [])}
    fork = {point.valid_time: point for point in intensity.get("fork", [])}
    starts = [point.valid_time for point in baseline.values() if point.step == 0]
    if not starts:
        return None
    start = starts[0]
    end = start + timedelta(hours=24)
    if any(time not in baseline or time not in fork for time in (start, end)):
        return None
    return (
        fork[end].min_mslp_pa
        - baseline[end].min_mslp_pa
        - fork[start].min_mslp_pa
        + baseline[start].min_mslp_pa
    ) / PA_PER_HPA


def compute_divergence(
    tracks: Mapping[RunName, pd.DataFrame],
    *,
    grid_resolution_deg: float = 1.0,
    mslp: PressureFields | None = None,
) -> Divergence:
    """Compare already matched tracks without interpolating missing positions.

    I use meridional grid spacing for the noise floor, independent of latitude.
    Pressure RMSE compares ensemble-mean fields with equal weight per grid cell
    inside 1500 km of the baseline mean. Without pressure fields it stays null.
    Intensity averages all available matched members, even below the five-member
    threshold for a published position. Drift needs exact 0 h and 24 h samples.

    Raises:
        ValueError: Tracks, grid spacing or supplied pressure fields are invalid.
    """
    if "baseline" not in tracks or not 0 < grid_resolution_deg <= MAX_GRID_SPACING_DEG:
        msg = "baseline tracks and grid_resolution_deg in (0, 180] are required"
        raise ValueError(msg)
    means = {name: ensemble_mean(frame).set_index("step") for name, frame in tracks.items()}
    times = pd.concat(
        [frame[["step", "valid_time"]] for frame in tracks.values()]
    ).drop_duplicates()
    if times.step.duplicated().any() or times.valid_time.duplicated().any():
        msg = "runs must share a one-to-one step and valid_time mapping"
        raise ValueError(msg)
    intensity: dict[RunName, list[IntensityPoint]] = {}
    for name, frame in tracks.items():
        values = frame[["mslp_pa", "wind_ms"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            msg = "track pressures and wind speeds must be finite and nonnegative"
            raise ValueError(msg)
        intensity[name] = [
            IntensityPoint.model_validate(
                {
                    "step": step,
                    "valid_time": time,
                    "count": len(group),
                    "min_mslp_pa": float(group.mslp_pa.mean()),
                    "max_wind_ms": float(group.wind_ms.mean()),
                }
            )
            for (step, time), group in frame.groupby(["step", "valid_time"], sort=True)
        ]
    rows: list[DivergenceStep] = []
    grid_cell_km = float(np.deg2rad(grid_resolution_deg) * EARTH_RADIUS_KM)
    for _, time in times.sort_values("step").iterrows():
        step = int(time.step)
        positions: dict[RunName, pd.DataFrame] = {
            name: frame.loc[frame.step == step] for name, frame in tracks.items()
        }
        count, pvalue = _paired_distances(positions)
        row = DivergenceStep(
            step=step, valid_time=time.valid_time, paired_members=count, paired_p=pvalue
        )
        if step in means["baseline"].index:
            baseline = means["baseline"].loc[[step]].iloc[0]
            members = positions["baseline"]
            row.baseline_spread_km = float(
                great_circle_km(baseline.lat, baseline.lon, members.lat, members.lon).mean()
            )
            separations = {
                name: float(
                    great_circle_km(
                        baseline.lat, baseline.lon, mean.loc[step].lat, mean.loc[step].lon
                    )
                )
                for name, mean in means.items()
                if step in mean.index
            }
            row.fork_sep_km, row.control_sep_km = (
                separations.get("fork"),
                separations.get("control"),
            )
            if row.fork_sep_km is not None and row.control_sep_km is not None:
                row.signal_to_noise = row.fork_sep_km / max(row.control_sep_km, grid_cell_km)
            if mslp is not None:
                row.mslp_rmse_pa = mslp.rmse_pa(step, float(baseline.lat), float(baseline.lon))
        rows.append(row)
    return Divergence(
        grid_resolution_deg=grid_resolution_deg,
        steps=rows,
        intensity=intensity,
        intensity_drift_24h_hpa=_drift(intensity),
    )
