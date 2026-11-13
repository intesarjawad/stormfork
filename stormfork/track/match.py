import numpy as np
import pandas as pd

from stormfork.grid import great_circle_km

INITIAL_RADIUS_KM = 200.0
GENESIS_RADIUS_KM = 500.0
MIN_COMMON_TIMES = 4
MAX_GAP_STEPS = 1  # Missing six-hourly fixes a track may skip and still continue.
MIN_MEAN_MEMBERS = 5  # A published eight-member ensemble needs a majority.
MAX_LATITUDE_DEG = 90.0
LONGITUDE_PERIOD_DEG = 360.0
TRACK_COLUMNS = ("member", "step", "valid_time", "lat", "lon", "mslp_pa", "wind_ms", "track_id")
MEAN_COLUMNS = ("step", "valid_time", "lat", "lon", "mslp_pa", "wind_ms", "count")


def _check_positions(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        msg = f"track table is missing columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    positions = frame[["lat", "lon"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(positions)) or np.any(abs(positions[:, 0]) > MAX_LATITUDE_DEG):
        msg = "track positions must be finite degrees with latitude in [-90, 90]"
        raise ValueError(msg)
    if not frame.empty and (
        frame.valid_time.isna().any() or not pd.api.types.is_datetime64_any_dtype(frame.valid_time)
    ):
        msg = "valid_time must contain non-missing datetimes"
        raise ValueError(msg)


def _check_tracks(tracks: pd.DataFrame) -> None:
    _check_positions(tracks, TRACK_COLUMNS)
    if tracks[["member", "step", "track_id"]].isna().any().any():
        msg = "member, step and track_id must not be missing"
        raise ValueError(msg)
    indices = tracks[["member", "step"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(indices)) or np.any(indices < 0) or np.any(indices != indices // 1):
        msg = "member and step must be nonnegative integers"
        raise ValueError(msg)
    if (
        tracks.duplicated(["member", "track_id", "valid_time"]).any()
        or tracks.duplicated(["member", "track_id", "step"]).any()
    ):
        msg = "a track must have one position per step and valid_time"
        raise ValueError(msg)
    times = tracks[["step", "valid_time"]].drop_duplicates()
    if times.step.duplicated().any() or times.valid_time.duplicated().any():
        msg = "step and valid_time must have a one-to-one mapping"
        raise ValueError(msg)


def extrapolated(path: pd.DataFrame, step: int) -> tuple[float, float]:
    """Return the (lat, lon) a path reaches at a step by moving at its last fix-to-fix velocity."""
    last = path.iloc[-1]
    if len(path) < 2:  # noqa: PLR2004
        return float(last.lat), float(last.lon)
    before = path.iloc[-2]
    scale = (step - last.step) / (last.step - before.step)
    wrap = (last.lon - before.lon + 180.0) % LONGITUDE_PERIOD_DEG - 180.0
    lat = min(max(last.lat + (last.lat - before.lat) * scale, -MAX_LATITUDE_DEG), MAX_LATITUDE_DEG)
    return float(lat), float((last.lon + wrap * scale) % LONGITUDE_PERIOD_DEG)


def _continue_across_gaps(member: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    """Join tracks the tracker restarted under a new id after missing up to one fix."""
    used = {path.track_id.iloc[0]}
    while True:
        last_step = int(path.step.iloc[-1])
        best, best_distance = None, INITIAL_RADIUS_KM
        for track_id, candidate in member.groupby("track_id", sort=True):
            first = candidate.iloc[0]
            if track_id in used or not 1 <= first.step - last_step <= MAX_GAP_STEPS + 1:
                continue
            lat, lon = extrapolated(path, int(first.step))
            distance = float(great_circle_km(lat, lon, first.lat, first.lon))
            if distance <= best_distance:
                best, best_distance = candidate, distance
        if best is None:
            return path
        used.add(best.track_id.iloc[0])
        path = pd.concat([path, best.assign(track_id=path.track_id.iloc[0])])


def _initial_match(member: pd.DataFrame, position: tuple[float, float]) -> pd.DataFrame:
    starts = member.loc[member.step == 0]
    distances = great_circle_km(position[0], position[1], starts.lat, starts.lon)
    ids = starts.loc[distances <= INITIAL_RADIUS_KM, "track_id"]
    if len(ids) > 1:
        msg = "multiple tracks lie within 200 km of the initial storm position"
        raise ValueError(msg)
    path = member.loc[member.track_id.isin(ids)]
    return _continue_across_gaps(member, path) if len(path) else path


def _genesis_match(member: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    best = member.iloc[:0]
    best_distance = float("inf")
    tied = False
    for _, predicted in member.groupby("track_id", sort=True):
        common = predicted.merge(observed[["valid_time", "lat", "lon"]], on="valid_time")
        if len(common) < MIN_COMMON_TIMES:
            continue
        distances = great_circle_km(common.lat_x, common.lon_x, common.lat_y, common.lon_y)
        if np.any(distances > GENESIS_RADIUS_KM):
            continue
        mean_distance = float(distances.mean())
        if mean_distance < best_distance:
            best, best_distance, tied = predicted, mean_distance, False
        elif mean_distance == best_distance:
            tied = True
    if tied:
        msg = "multiple tracks have the same minimum mean distance to the observed storm"
        raise ValueError(msg)
    return best


def match_storm(
    tracks: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    initial_position: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Select one target track per member, retaining its entire predicted path.

    Tables use lat/lon degrees and datetime valid_time. Tracks also carry member,
    step, mslp_pa (Pa), wind_ms (m/s) and track_id. For a storm present at init,
    supply its initial position, moved for a relocation: only step 0 is matched
    within 200 km, and a track the tracker restarted under a new id after missing
    at most one six-hourly fix continues it when its first fix lies within 200 km
    of the position extrapolated from the track's last two fixes. Otherwise,
    match at least four common observation times, all within 500 km, choosing the
    smallest mean great-circle distance. Missing members contribute no rows.
    No observations are needed in the initial case.

    Raises:
        ValueError: Tables are invalid or storm identity is ambiguous.
    """
    _check_tracks(tracks)
    if initial_position is None:
        _check_positions(observed, ("valid_time", "lat", "lon"))
        if observed.valid_time.duplicated().any():
            msg = "observed storm must have one position per valid_time"
            raise ValueError(msg)
    elif not np.all(np.isfinite(initial_position)) or abs(initial_position[0]) > MAX_LATITUDE_DEG:
        msg = "initial position must be finite degrees with latitude in [-90, 90]"
        raise ValueError(msg)
    selected = [
        _initial_match(member, initial_position)
        if initial_position is not None
        else _genesis_match(member, observed)
        for _, member in tracks.groupby("member", sort=True)
    ]
    if not selected:
        return tracks.iloc[:0].copy()
    return pd.concat(selected).sort_values(["member", "step"]).reset_index(drop=True)


def ensemble_mean(matched: pd.DataFrame) -> pd.DataFrame:
    """Average matched tracks at steps with at least five distinct members.

    Return step, valid_time, lat/lon degrees, mslp_pa, wind_ms and count. Latitude
    and intensities use arithmetic means; longitude uses a circular mean in
    [0, 360), so tracks crossing the seam remain together. Missing positions are
    never interpolated. Input must contain at most one target track per member.

    Raises:
        ValueError: Tracks are invalid or include multiple storms per member.
    """
    _check_tracks(matched)
    if matched[["member", "track_id"]].drop_duplicates().member.duplicated().any():
        msg = "ensemble mean requires only one matched track per member"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for (step, valid_time), positions in matched.groupby(["step", "valid_time"], sort=True):
        count = len(positions)
        if count < MIN_MEAN_MEMBERS:
            continue
        longitude = np.deg2rad(positions.lon.to_numpy(dtype=float))
        mean_lon = float(
            np.rad2deg(np.arctan2(np.sin(longitude).mean(), np.cos(longitude).mean()))
            % LONGITUDE_PERIOD_DEG
        )
        # A negative roundoff residual can wrap to exactly 360 degrees.
        if mean_lon == LONGITUDE_PERIOD_DEG:
            mean_lon = 0.0
        rows.append(
            {
                "step": step,
                "valid_time": valid_time,
                "lat": float(positions.lat.mean()),
                "lon": mean_lon,
                "mslp_pa": float(positions.mslp_pa.mean()),
                "wind_ms": float(positions.wind_ms.mean()),
                "count": count,
            }
        )
    return pd.DataFrame(rows, columns=list(MEAN_COLUMNS))
