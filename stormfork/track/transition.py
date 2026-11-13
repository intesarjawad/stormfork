import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

from stormfork.grid import STEP, great_circle_km
from stormfork.track.match import TRACK_COLUMNS, extrapolated

SEARCH_RADIUS_KM = 400.0  # Beyond the 6 h drift of a recurving storm plus a wobble in its last fix.
CLOSED_CONTOUR_PA = 200.0
CLOSED_CONTOUR_KM = 600.0  # 5.5 degrees of great-circle distance, rounded.
RING_WIDTH_KM = 60.0
OWN_LOW_KM = 200.0  # Identity tolerance; Sandy sensitivity is recorded in ADR 0008.
KM_PER_DEGREE = 111.0
METRES_PER_SECOND_PER_KNOT = 1852 / 3600
MSLP = "mean_sea_level_pressure"
USA_WIND = "cyclone_usa_wind_disc"


def _vertex(below: float, centre: float, above: float) -> float:
    """Offset in grid steps of the vertex of the parabola through three equally spaced values."""
    curvature = below - 2 * centre + above
    if not np.isfinite(curvature) or curvature <= 0:
        return 0.0
    return float(np.clip(0.5 * (below - above) / curvature, -0.5, 0.5))


def _closed_minima(
    mslp: xr.DataArray, lat: float, lon: float, *, radius_km: float = SEARCH_RADIUS_KM
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the lat, lon and pressure of minima near a position that have a closed contour.

    A minimum is closed when the mean pressure on a ring 5.5 degrees away exceeds it by
    200 Pa, the TempestExtremes DetectNodes criterion for extratropical lows. Each grid-point
    minimum is refined to the vertex of a parabola through it and its neighbours along each
    axis, so positions carry sub-grid precision like the tracker's own fixes.
    """
    lats, lons = mslp.lat.to_numpy(), mslp.lon.to_numpy()
    spacing = float(lons[1] - lons[0])
    grid_diagonal_km = spacing * KM_PER_DEGREE * np.sqrt(2)
    reach = (radius_km + grid_diagonal_km + CLOSED_CONTOUR_KM + RING_WIDTH_KM) / KM_PER_DEGREE
    lat_index = np.flatnonzero(np.abs(lats - lat) <= reach)
    lon_reach = min(reach / max(np.cos(np.deg2rad(min(abs(lat) + reach, 89.0))), 0.05), 180.0)
    centre = round(lon / spacing)
    lon_index = (
        centre + np.arange(-int(lon_reach / spacing), int(lon_reach / spacing) + 1)
    ) % lons.size
    values = mslp.to_numpy()[np.ix_(lat_index, lon_index)]
    grid_lat = np.broadcast_to(lats[lat_index][:, None], values.shape)
    grid_lon = np.broadcast_to(lons[lon_index][None, :], values.shape)
    lat_spacing = float(lats[1] - lats[0])
    padded = np.pad(values, 1, constant_values=np.inf)
    neighbours = np.stack(
        [
            padded[1 + i : 1 + i + values.shape[0], 1 + j : 1 + j + values.shape[1]]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        ]
    )
    near = great_circle_km(lat, lon, grid_lat, grid_lon) <= radius_km + grid_diagonal_km
    found: list[tuple[float, float, float]] = []
    for row, col in zip(*np.nonzero(near & (values < neighbours.min(axis=0))), strict=True):
        away = great_circle_km(grid_lat[row, col], grid_lon[row, col], grid_lat, grid_lon)
        ring = np.abs(away - CLOSED_CONTOUR_KM) <= RING_WIDTH_KM
        if ring.any() and values[ring].mean() - values[row, col] >= CLOSED_CONTOUR_PA:
            centre_value = float(padded[row + 1, col + 1])
            along_lat = _vertex(
                float(padded[row, col + 1]), centre_value, float(padded[row + 2, col + 1])
            )
            along_lon = _vertex(
                float(padded[row + 1, col]), centre_value, float(padded[row + 1, col + 2])
            )
            found.append(
                (
                    float(grid_lat[row, col]) + along_lat * lat_spacing,
                    (float(grid_lon[row, col]) + along_lon * spacing) % 360.0,
                    float(centre_value),
                )
            )
    lows = np.array(found).reshape(-1, 3)
    lows = lows[great_circle_km(lat, lon, lows[:, 0], lows[:, 1]) <= radius_km]
    return lows[:, 0], lows[:, 1], lows[:, 2]


def _lost_identity(
    candidate: tuple[float, float],
    last: tuple[float, float],
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    """Reject an unidentifiable handover or a candidate descended from a deeper neighbour.

    Match distinct lows one-to-one before treating unmatched sources as a merger.
    The own low may move faster than 400 km if it passes the extrapolation test.
    """
    lats, lons, pressures = before
    if lats.size == 0:
        return True
    current_lats, current_lons, _ = current
    distances = great_circle_km(lats[:, None], lons[:, None], current_lats, current_lons)
    destination = int(np.argmin(great_circle_km(*candidate, current_lats, current_lons)))
    from_last = great_circle_km(last[0], last[1], lats, lons)
    is_own = from_last <= OWN_LOW_KM
    allowed = distances <= SEARCH_RADIUS_KM
    allowed[:, destination] |= is_own
    # Costlier than every valid pair combined, so invalid pairs cannot displace valid ones.
    penalty = (float(distances.max()) + 1) * (min(distances.shape) + 1)
    sources, destinations = linear_sum_assignment(np.where(allowed, distances, penalty))
    assigned = np.full(lats.size, -1, dtype=int)
    for source, target in zip(sources, destinations, strict=True):
        if allowed[source, target]:
            assigned[source] = target
    linked = (
        (assigned == destination) | ((assigned < 0) & (np.argmin(distances, axis=1) == destination))
    ) & allowed[:, destination]
    own = pressures[is_own]
    if own.size == 0:
        return bool(
            not np.any(linked & (from_last <= SEARCH_RADIUS_KM))
            or np.any(linked & (from_last > SEARCH_RADIUS_KM))
        )
    own_index = int(np.flatnonzero(is_own)[np.argmin(own)])
    if assigned[own_index] >= 0 and assigned[own_index] != destination:
        return True
    return bool(np.any(linked & ~is_own & (pressures < own.min())))


def continue_through_transition(
    matched: pd.DataFrame, fields: xr.Dataset, init_time: np.datetime64
) -> pd.DataFrame:
    """Extend each member's matched track for as long as a closed pressure minimum follows it.

    The tracker ends a track when the model's cyclone-existence probability falls, which
    happens as a hurricane turns extratropical while its low persists. From the track's
    last fix, each next step takes the closed minimum of mean sea level pressure nearest
    the position extrapolated from the last two fixes, within 400 km, and stops when
    there is none, or when that minimum belongs to a deeper neighbouring low that had already
    closed at the previous step (the storm has filled into it, and following on would hand
    its track to another system). Pressure is the low's own grid minimum of mean sea level
    pressure: the USA pressure field describes a tropical cyclone and reads far deeper than
    the low over a decaying one. Wind still comes from the USA wind field the tracker reads,
    interpolated at the new position.

    Raises:
        ValueError: The fields lack mean sea level pressure.
    """
    if MSLP not in fields.data_vars:
        msg = f"tracker fields are missing {MSLP}, needed to follow storms that turn extratropical"
        raise ValueError(msg)
    last_step = int(fields["time"].to_numpy()[-1] / np.timedelta64(STEP))
    extended = [matched]
    for member, path in matched.groupby("member", sort=True):
        path = path.sort_values("step")  # noqa: PLW2901
        while int(path.step.iloc[-1]) < last_step:
            step = int(path.step.iloc[-1]) + 1
            lat, lon = extrapolated(path, step)
            now = fields.sel(member=member, time=step * np.timedelta64(STEP))
            lats, lons, pressures = _closed_minima(now[MSLP], lat, lon)
            if lats.size == 0:
                break
            nearest = int(np.argmin(great_circle_km(lat, lon, lats, lons)))
            last = (float(path.lat.iloc[-1]), float(path.lon.iloc[-1]))
            candidate = (float(lats[nearest]), float(lons[nearest]))
            previous = (step - 1) * np.timedelta64(STEP)
            if previous not in fields["time"].to_numpy():
                break
            radius = max(float(great_circle_km(*candidate, *last)) + OWN_LOW_KM, SEARCH_RADIUS_KM)
            before = _closed_minima(
                fields.sel(member=member, time=previous)[MSLP],
                *candidate,
                radius_km=2 * radius + SEARCH_RADIUS_KM,
            )
            # Include prior counterparts for every current competitor, plus its allowed drift.
            current = _closed_minima(now[MSLP], *candidate, radius_km=2 * radius)
            if _lost_identity(candidate, last, before, current):
                break
            seam = now[[USA_WIND]].isel(lon=[0])
            seam.coords["lon"] = [360.0]
            cyclic_wind = xr.concat([now[USA_WIND], seam[USA_WIND]], dim="lon")
            wind_ms = float(cyclic_wind.interp(lat=candidate[0], lon=candidate[1]))
            row = pd.DataFrame(
                {
                    "member": [member],
                    "step": [step],
                    "valid_time": [pd.Timestamp(init_time) + step * STEP],
                    "lat": [float(lats[nearest])],
                    "lon": [float(lons[nearest])],
                    "mslp_pa": [float(pressures[nearest])],
                    "wind_ms": [wind_ms * METRES_PER_SECOND_PER_KNOT],
                    "track_id": [path.track_id.iloc[0]],
                }
            )
            path = pd.concat([path, row])  # noqa: PLW2901
            extended.append(row)
    return (
        pd.concat(extended)
        .sort_values(["member", "step"])
        .reset_index(drop=True)
        .loc[:, list(TRACK_COLUMNS)]
    )
