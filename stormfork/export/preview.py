from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

SPATIAL_DIMENSIONS = 2


def write_grayscale(pixels: NDArray[np.generic], destination: Path) -> None:
    """Write lossless 8-bit L PNGs without colour profiles, gamma or alpha.

    Raises:
        ValueError: Pixels are not a two-dimensional uint8 array.
    """
    if pixels.ndim != SPATIAL_DIMENSIONS or pixels.dtype != np.uint8:
        msg = "PNG pixels must be a two-dimensional uint8 array"
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination, format="PNG", optimize=True)
