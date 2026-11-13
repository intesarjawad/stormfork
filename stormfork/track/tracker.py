from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import xarray as xr

from stormfork.grid import STEP, Grid
from stormfork.track.match import MAX_LATITUDE_DEG, TRACK_COLUMNS, match_storm
from stormfork.track.transition import continue_through_transition

PA_PER_HPA = 100.0
METRES_PER_SECOND_PER_KNOT = 1852 / 3600  # International nautical mile per hour.
QUADRANTS = ("ne", "se", "sw", "nw")
SCALAR_FIELDS = (
    "cyclone_usa_pres_disc",
    "cyclone_usa_wind_disc",
    "cyclone_usa_rmw_disc",
    *(
        f"cyclone_usa_r{speed}_{quadrant}_radius_disc"
        for speed in (34, 50, 64)
        for quadrant in QUADRANTS
    ),
)
GALE_RADIUS_COLUMNS = tuple(f"radius_34_knot_winds_{quadrant}_km" for quadrant in QUADRANTS)
# The IBTrACS preparation divides the recorded nautical miles by this to get kilometres.
NAUTICAL_MILES_PER_KM = 0.539957


class _DirectTracker(Protocol):
    def preprocess_gridded_ds(self, gridded_ds: xr.Dataset) -> xr.Dataset: ...

    def __call__(
        self, gridded_ds: xr.Dataset, initial_storms_df: pd.DataFrame, *, do_cyclogenesis: bool
    ) -> pd.DataFrame: ...


class _TrackerConfig(Protocol):
    tracker_constructor: Callable[..., _DirectTracker]
    tracker_kwargs: dict[str, object]


def _load_backend() -> tuple[
    Callable[[], _TrackerConfig],
    Callable[[xr.Dataset, np.datetime64], tuple[pd.DataFrame, pd.DataFrame]],
]:
    try:
        config_module = import_module("weathernext.cyclones.direct_tracker_6h_v1_config")
        ibtracs_module = import_module("weathernext.cyclones.ibtracs_netcdf_to_csv")
    except ImportError as error:
        msg = (
            f"WeatherNext cyclone tracker is unavailable ({error}). "
            "Install the pinned model dependencies with uv sync --group model."
        )
        raise ImportError(msg) from error
    return config_module.get_config, ibtracs_module.prepare_ibtracs_storms_dfs


def _whole_seconds(ibtracs: xr.Dataset) -> xr.Dataset:
    # NCEI stores times as float days since 1858, which decode up to tens of
    # microseconds off the hour, and the upstream filter matches the init exactly.
    return ibtracs.assign(time=ibtracs["time"].dt.round("s"))


def _position_at(initial: pd.DataFrame, init_time: np.datetime64, track_id: str) -> pd.DataFrame:
    current = initial.loc[
        (initial.track_id == track_id) & (initial.valid_time == pd.Timestamp(init_time))
    ]
    if len(current) != 1:
        msg = f"IBTrACS holds no single initial position for storm {track_id} at {init_time}"
        raise ValueError(msg)
    return current


def _move_initial_storm(
    initial: pd.DataFrame,
    init_time: np.datetime64,
    track_id: str,
    destination: tuple[float, float],
) -> pd.DataFrame:
    target = initial.track_id == track_id
    current = _position_at(initial, init_time, track_id)
    if not np.all(np.isfinite(destination)) or abs(destination[0]) > MAX_LATITUDE_DEG:
        msg = "moved initial position must be finite degrees with latitude in [-90, 90]"
        raise ValueError(msg)
    moved = initial.copy(deep=True)
    moved.loc[target, "lat"] += destination[0] - float(current.lat.iloc[0])
    moved.loc[target, "lon"] = (
        moved.loc[target, "lon"] + destination[1] - float(current.lon.iloc[0])
    ) % 360
    if (abs(moved.loc[target, "lat"]) > MAX_LATITUDE_DEG).any():
        msg = f"relocation places storm {track_id} history outside latitude [-90, 90]"
        raise ValueError(msg)
    return moved


def _prepare_fields(fields: xr.Dataset, init_time: np.datetime64) -> xr.Dataset:
    Grid.from_dataset(fields)
    missing = set(SCALAR_FIELDS) - set(fields.data_vars)
    if missing:
        msg = f"tracker fields are missing cyclone outputs: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    if np.isnat(init_time):
        msg = "init_time must be a non-missing datetime"
        raise ValueError(msg)
    if "time" not in fields.coords:
        msg = "tracker fields require a time coordinate of six-hour lead timedeltas"
        raise ValueError(msg)
    leads = fields["time"].to_numpy()
    interval = np.timedelta64(STEP)
    if (
        leads.ndim != 1
        or leads.size == 0
        or not np.issubdtype(leads.dtype, np.timedelta64)
        or np.any(np.isnat(leads))
    ):
        msg = "time must be a nonempty one-dimensional coordinate of lead timedeltas"
        raise ValueError(msg)
    if (
        leads[0] not in (np.timedelta64(0, "h"), interval)
        or np.any(np.diff(leads) != interval)
        or leads[-1] < interval
    ):
        msg = "lead times must start at 0 or 6 hours, advance by 6 hours and include a forecast"
        raise ValueError(msg)
    prepared = fields
    if "batch" in prepared.dims:
        if prepared.sizes["batch"] != 1:
            msg = "tracker expects a single initialization (batch size 1)"
            raise ValueError(msg)
        prepared = prepared.isel(batch=0, drop=True)
    prepared = prepared.copy()
    prepared.coords["init_time"] = init_time
    return prepared


def _normalise(raw: pd.DataFrame, member: int, init_time: np.datetime64) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=list(TRACK_COLUMNS))
    result = raw.rename(
        columns={
            "minimum_sea_level_pressure_hpa": "mslp_pa",
            "maximum_sustained_wind_speed_knots": "wind_ms",
        }
    ).copy()
    result["member"] = member
    result["valid_time"] = pd.to_datetime(result.valid_time)
    lead_steps = (result.valid_time - pd.Timestamp(init_time)) / pd.Timedelta(STEP)
    if np.any(lead_steps < 0) or np.any(lead_steps != np.floor(lead_steps)):
        msg = "tracker returned valid times outside the six-hour forecast steps"
        raise ValueError(msg)
    result["step"] = lead_steps.astype(int)
    result["lon"] %= 360
    result["mslp_pa"] *= PA_PER_HPA
    result["wind_ms"] *= METRES_PER_SECOND_PER_KNOT
    result["track_id"] = result.track_id.astype(str)
    return result.loc[:, list(TRACK_COLUMNS)]


class StormStart(NamedTuple):
    """The target storm at an init, as IBTrACS records it at six-hourly times.

    position is (lat, lon) in degrees with longitude in [0, 360), or None when
    the init precedes first_record_time, the storm's first record (naive UTC).
    gale_radii_nm holds the USA 34 kt radius of each quadrant (NE, SE, SW, NW)
    at the init in nautical miles, NaN where unrecorded, and is empty without
    a position.
    """

    position: tuple[float, float] | None
    first_record_time: datetime
    gale_radii_nm: tuple[float, ...] = ()


def storm_start(ibtracs: xr.Dataset, init_time: np.datetime64, track_id: str) -> StormStart:
    """Return a storm's IBTrACS position at the init, as the tracker reads it, and its first record.

    Raises:
        ImportError: The optional WeatherNext cyclone dependencies are missing.
        ValueError: IBTrACS has no record of the storm, or no single position
            at an init after its first record.
    """
    _, prepare_ibtracs = _load_backend()
    observed, initial = prepare_ibtracs(_whole_seconds(ibtracs), init_time)
    records = observed.loc[observed.track_id == track_id, "valid_time"]
    if records.empty:
        msg = f"IBTrACS holds no record of storm {track_id}"
        raise ValueError(msg)
    first_record_time = pd.Timestamp(records.min()).to_pydatetime()
    if first_record_time > pd.Timestamp(init_time):
        return StormStart(None, first_record_time)
    current = _position_at(initial, init_time, track_id)
    gale_radii_nm = tuple(
        float(current[column].iloc[0]) * NAUTICAL_MILES_PER_KM for column in GALE_RADIUS_COLUMNS
    )
    return StormStart(
        (float(current.lat.iloc[0]), float(current.lon.iloc[0])), first_record_time, gale_radii_nm
    )


def track(
    fields: xr.Dataset,
    ibtracs: xr.Dataset,
    init_time: np.datetime64,
    *,
    target_track_id: str | None = None,
    moved_initial_position: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Track six-hour WeatherNext cyclone fields with cyclogenesis enabled.

    Fields use time lead timedeltas, lat/lon degrees on a regular global-longitude
    grid, optional member labels (nonnegative integers), and optional singleton
    batch. Include cyclone existence, USA pressure/wind and all USA radius fields.
    IBTrACS is an already loaded NetCDF dataset; this function performs no I/O.
    Initialization comes from IBTrACS at init_time, including its lookback history.
    For relocation, supply the target's ID and moved (lat, lon) in degrees: its
    history receives the same displacement, leaving other storms unchanged.

    Return member, step (0 at init), valid_time, lat, lon in [0, 360), mslp_pa,
    wind_ms and track_id. Pressure and wind convert from hPa and knots to Pa and
    m/s. WeatherNext's default minimum genesis duration remains in effect.
    Without a target, every storm the tracker follows is returned. With one,
    each member keeps only the track that starts within 200 km of the target's
    initial position, moved for relocation; a member that lost it has no rows.
    A target whose first IBTrACS record follows the init forms during the
    forecast, so each member keeps the track nearest its observed path instead
    (at least four common valid times, all within 500 km). The kept track then
    continues for as long as a closed minimum of mean sea level pressure follows
    it, because the tracker ends tracks when a storm turns extratropical.

    Raises:
        ImportError: The optional WeatherNext cyclone dependencies are missing.
        ValueError: Grid, lead times, members or relocation are invalid, the
            target is absent from IBTrACS at the init, or its match is ambiguous.
    """
    if moved_initial_position is not None and target_track_id is None:
        msg = "target_track_id is required with a moved initial position"
        raise ValueError(msg)
    prepared = _prepare_fields(fields, init_time)
    get_config, prepare_ibtracs = _load_backend()
    observed, initial = prepare_ibtracs(_whole_seconds(ibtracs), init_time)
    if moved_initial_position is not None and target_track_id is not None:
        initial = _move_initial_storm(initial, init_time, target_track_id, moved_initial_position)
    if "member" not in prepared.dims:
        prepared = prepared.expand_dims(member=[0])
    members = np.asarray(prepared.member.values, dtype=float)
    if (
        not np.all(np.isfinite(members))
        or np.any(members < 0)
        or np.any(members != np.floor(members))
        or len(np.unique(members)) != len(members)
        or members.size == 0
    ):
        msg = "member labels must be distinct nonnegative integers"
        raise ValueError(msg)
    results: list[pd.DataFrame] = []
    for index, member in enumerate(members):
        config = get_config()
        tracker = config.tracker_constructor(**config.tracker_kwargs)
        member_fields = tracker.preprocess_gridded_ds(prepared.isel(member=index, drop=True))
        raw = tracker(member_fields, initial.copy(deep=True), do_cyclogenesis=True)
        results.append(_normalise(raw, int(member), init_time))
    tracks = pd.concat(results, ignore_index=True).sort_values(["member", "step", "track_id"])
    if target_track_id is None:
        return tracks
    target_observed = observed.loc[observed.track_id == target_track_id]
    forms_after_init = (target_observed.valid_time > pd.Timestamp(init_time)).all()
    if moved_initial_position is None and not target_observed.empty and forms_after_init:
        matched = match_storm(tracks, target_observed.loc[:, ["valid_time", "lat", "lon"]])
        return continue_through_transition(matched, prepared, init_time)
    target = _position_at(initial, init_time, target_track_id)
    position = (float(target.lat.iloc[0]), float(target.lon.iloc[0]))
    matched = match_storm(tracks, pd.DataFrame(), initial_position=position)
    return continue_through_transition(matched, prepared, init_time)
