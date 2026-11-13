from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stormfork.export.preview import write_grayscale
from stormfork.export.quantise import FILE_NAMES, FieldMetadata, field_metadata, quantise

FRAME_COUNT = 41
ATLAS_COLUMNS = 5
ATLAS_ROWS = 9
LATITUDE_COUNT = 181
LONGITUDE_COUNT = 360
FRAME_SHAPE = (FRAME_COUNT, LATITUDE_COUNT, LONGITUDE_COUNT)
RESOLUTIONS_DEG = (1, 2)


def tile_origin(frame: int, *, resolution_deg: int = 1) -> tuple[int, int]:
    """Return a frame's top-left (x, y) pixel, with latitude ascending in rows.

    Raises:
        ValueError: The frame is outside 0..40 or resolution is not 1°/2°.
    """
    if not 0 <= frame < FRAME_COUNT:
        msg = "frame must be in [0, 40]"
        raise ValueError(msg)
    if resolution_deg not in RESOLUTIONS_DEG:
        msg = "resolution_deg must be 1 or 2"
        raise ValueError(msg)
    return (
        frame % ATLAS_COLUMNS * (LONGITUDE_COUNT // resolution_deg + 1),
        frame // ATLAS_COLUMNS * ((LATITUDE_COUNT - 1) // resolution_deg + 1),
    )


def pack_atlas(frames: NDArray[np.generic], *, resolution_deg: int = 1) -> NDArray[np.uint8]:
    """Tile 1 to 41 byte frames, repeating longitude zero at each tile's east edge.

    Frames must already be sampled at resolution_deg, ordered (frame, lat, lon),
    with latitudes -90..90 and longitudes 0..360-resolution_deg. The atlas keeps
    its full size whatever the frame count, so tiles after the last frame remain
    zero; the manifest's steps says how many hold frames. Sampling at 2° keeps
    both poles.

    Raises:
        ValueError: The byte-array shape or resolution is invalid.
    """
    tile_origin(0, resolution_deg=resolution_deg)
    height = (LATITUDE_COUNT - 1) // resolution_deg + 1
    width = LONGITUDE_COUNT // resolution_deg
    if (
        frames.ndim != len(FRAME_SHAPE)
        or not 1 <= frames.shape[0] <= FRAME_COUNT
        or frames.shape[1:] != (height, width)
        or frames.dtype != np.uint8
    ):
        msg = f"frames must be uint8 with shape (1 to {FRAME_COUNT}, {height}, {width})"
        raise ValueError(msg)
    atlas = np.zeros((ATLAS_ROWS * height, ATLAS_COLUMNS * (width + 1)), dtype=np.uint8)
    for frame in range(frames.shape[0]):
        x, y = tile_origin(frame, resolution_deg=resolution_deg)
        atlas[y : y + height, x : x + width] = frames[frame]
        atlas[y : y + height, x + width] = frames[frame, :, 0]
    return atlas


def export_run(
    fields: Mapping[str, NDArray[np.float64]],
    destination: Path,
    run: str,
    *,
    wind_resolution_deg: int = 1,
) -> list[FieldMetadata]:
    """Write six atlases at <run>/<file>.png and previews at preview/<run>_<file>_f0.png.

    Input is six derived fields in native units, each (frames, 181, 360) with the
    same 1 to 41 frames (frame 0 the analysis, then one per six-hour step), ordered
    south to north and 0..359° east. Ensemble reduction and z500 selection happen
    upstream. Precipitation's analysis is always encoded zero, even if missing.
    Wind fallback takes every second point without averaging. Previews retain
    the full 360x181 first frame even when wind atlases use 2° sampling.

    Raises:
        ValueError: Fields, shapes, finite values, run name or resolution are invalid.
    """
    metadata = field_metadata(wind_resolution_deg=wind_resolution_deg)
    if not run or run in (".", "..", "preview") or any(char in run for char in "/\\:"):
        msg = "run must be a single directory name other than preview"
        raise ValueError(msg)
    if set(fields) != set(FILE_NAMES):
        msg = f"fields must contain exactly {FILE_NAMES}"
        raise ValueError(msg)
    frame_count = np.shape(fields[FILE_NAMES[0]])[0]
    if not 1 <= frame_count <= FRAME_COUNT:
        msg = (
            f"an atlas holds 1 to {FRAME_COUNT} frames, the analysis and up to "
            f"{FRAME_COUNT - 1} six-hour steps; got {frame_count}"
        )
        raise ValueError(msg)
    expected = (frame_count, LATITUDE_COUNT, LONGITUDE_COUNT)
    encoded: dict[str, NDArray[np.uint8]] = {}
    for file in FILE_NAMES:
        values = np.asarray(fields[file], dtype=np.float64)
        if values.shape != expected:
            msg = f"{file}: expected shape {expected}, got {values.shape}"
            raise ValueError(msg)
        if file == "tp6h":
            values = values.copy()
            values[0] = 0
        encoded[file] = quantise(values, file)
    for file, frames in encoded.items():
        resolution = wind_resolution_deg if file in ("u10", "v10") else 1
        atlas = pack_atlas(frames[:, ::resolution, ::resolution], resolution_deg=resolution)
        write_grayscale(atlas, destination / run / f"{file}.png")
        write_grayscale(frames[0], destination / "preview" / f"{run}_{file}_f0.png")
    return metadata
