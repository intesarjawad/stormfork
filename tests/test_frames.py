from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from stormfork.export.frames import ATLAS_VARIABLES, atlas_fields
from stormfork.grid import LAT, LON

if TYPE_CHECKING:
    from numpy.typing import NDArray

LEVELS = [500, 850]


def dataset(times: int, members: int, value: float) -> xr.Dataset:
    """Every value encodes its member, time and level, so a wrong pick shows."""
    member = np.arange(members)[:, None, None, None, None] * 1000
    time = np.arange(times)[None, None, :, None, None] * 10
    surface = np.broadcast_to(member + time + value, (members, 1, times, LAT.size, LON.size))
    level = np.array(LEVELS)[None, None, None, :, None, None]
    upper = np.broadcast_to(
        surface[:, :, :, None] + level, (members, 1, times, len(LEVELS), LAT.size, LON.size)
    )
    variables: dict[str, tuple[tuple[str, ...], NDArray[np.generic]]] = {
        name: (("member", "batch", "time", "lat", "lon"), surface)
        for name in ATLAS_VARIABLES.values()
        if name != "geopotential"
    }
    variables["geopotential"] = (("member", "batch", "time", "level", "lat", "lon"), upper)
    return xr.Dataset(
        variables,
        coords={"time": np.arange(times), "level": LEVELS, "lat": LAT, "lon": LON},
    ).isel(member=slice(None) if members > 1 else 0)


def test_frame_zero_is_the_init_and_later_frames_are_member_zero():
    inputs = dataset(times=2, members=1, value=1)
    forecast = dataset(times=3, members=2, value=2)
    fields = atlas_fields(inputs, forecast)
    assert set(fields) == set(ATLAS_VARIABLES)
    for file, values in fields.items():
        assert values.shape == (4, LAT.size, LON.size)
        level = 500 if file == "z500" else 0
        expected_start = 0 if file == "tp6h" else 11 + level
        assert np.all(values[0] == expected_start)
        for step in range(3):
            assert np.all(values[step + 1] == 2 + step * 10 + level)
