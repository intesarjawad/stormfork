from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

# Distances use IUGG GRS80 mean radius R1=(2a+b)/3, rounded to 6371.0088 km.
# Moritz, Geodetic Reference System 1980: https://geoweb.mit.edu/~tah/12.221_2005/grs80_corr.pdf
from stormfork.grid import Grid, great_circle_km

# International nautical mile, IHB 1929 (IHO M-3 resolution 2/1919): exactly 1852 m.
# https://iho.int/iho_pubs/misc/M3-E-AUGUST18.pdf
KM_PER_NAUTICAL_MILE = 1.852
R34_RADIUS_FACTOR = 1.5  # Chosen margin around the largest reported gale-force quadrant.
MIN_RADIUS_KM = 400.0
MAX_RADIUS_KM = 900.0
R34_COLUMNS = ("USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW")
SPATIAL_DIMENSIONS = 2
POLE_LATITUDE_DEG = 90
IBTRACS_MISSING = -9999.0


def _ibtracs_number(value: str | float | None) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.nan
    return float(value)


def taper_radius_km(r34_nm: Sequence[float]) -> float | None:
    """The taper radius a storm's gale-force (34 kt) quadrant radii give, in km.

    The largest recorded quadrant, in nautical miles, times R34_RADIUS_FACTOR,
    clipped to [MIN_RADIUS_KM, MAX_RADIUS_KM]. Nonfinite and negative radii
    are missing; None when every quadrant is.
    """
    radii = np.asarray(r34_nm, dtype=np.float64)
    available = radii[np.isfinite(radii) & (radii >= 0)]
    if available.size == 0:
        return None
    return float(
        np.clip(
            R34_RADIUS_FACTOR * available.max() * KM_PER_NAUTICAL_MILE,
            MIN_RADIUS_KM,
            MAX_RADIUS_KM,
        )
    )


@dataclass(frozen=True)
class Taper:
    """Circular weight with a flat inner half and a cosine falloff to the edge.

    centre_lat and centre_lon are latitude/longitude in degrees; radius_km is
    positive kilometres. weights returns a dimensionless float64 (lat, lon)
    array in [0, 1] on the supplied regular Grid, independent of frame/level axes.
    The inner-half plateau is a chosen profile to leave the storm core intact.
    """

    centre_lat: float
    centre_lon: float
    radius_km: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.centre_lat)
            or not -POLE_LATITUDE_DEG <= self.centre_lat <= POLE_LATITUDE_DEG
        ):
            msg = "centre_lat must be finite and between -90 and 90 degrees"
            raise ValueError(msg)
        if not np.isfinite(self.centre_lon):
            msg = "centre_lon must be finite, in degrees"
            raise ValueError(msg)
        if not np.isfinite(self.radius_km) or self.radius_km <= 0:
            msg = "radius_km must be finite and positive"
            raise ValueError(msg)

    @classmethod
    def from_ibtracs(cls, row: Mapping[str, str | float | None]) -> "Taper":
        """Use one storm's init-time row, with USA wind radii in nautical miles.

        Blank, absent, nonfinite and negative R34 values are missing. Zero is a
        valid quadrant radius. Positions in degrees reject the -9999 sentinel.
        The row must already be selected for SID and time.

        Raises:
            ValueError: All R34 quadrants are missing, or the centre is invalid.
        """
        radius_km = taper_radius_km([_ibtracs_number(row.get(column)) for column in R34_COLUMNS])
        if radius_km is None:
            msg = f"{row.get('SID')} at {row.get('ISO_TIME')}: all USA_R34 quadrants are missing"
            raise ValueError(msg)
        for column in ("USA_LAT", "USA_LON"):
            if _ibtracs_number(row.get(column)) == IBTRACS_MISSING:
                msg = f"{column}: missing IBTrACS position (-9999)"
                raise ValueError(msg)
        return cls(
            _ibtracs_number(row.get("USA_LAT")),
            _ibtracs_number(row.get("USA_LON")),
            radius_km,
        )

    def weights(self, grid: Grid) -> NDArray[np.float64]:
        """Return (lat, lon) weights using great-circle distances in km."""
        distance_km = great_circle_km(self.centre_lat, self.centre_lon, grid.lat[:, None], grid.lon)
        transition = np.clip(2 * (distance_km / self.radius_km) - 1, 0, 1)
        return (1 + np.cos(np.pi * transition)) / 2


def background(field: ArrayLike, sigma_grid_cells: float = 6.0) -> NDArray[np.float64]:
    """Smooth each final (lat, lon) slice with sigma_grid_cells in grid cells.

    The default six-cell width is a chosen background scale, not a physical constant.
    Longitude wraps and latitude reflects at the boundary. Floating-point output
    prevents integer inputs from truncating the Gaussian convolution. The filter
    of defined values is divided by the filter of their mask, so missing cells
    do not dilute the background. NaNs remain exactly at the input NaN locations.

    Raises:
        ValueError: Spatial axes are missing or sigma_grid_cells is not finite and nonnegative.
    """
    values = np.asarray(field, dtype=np.float64)
    if values.ndim < SPATIAL_DIMENSIONS:
        msg = "field must have final lat and lon axes"
        raise ValueError(msg)
    if not np.isfinite(sigma_grid_cells) or sigma_grid_cells < 0:
        msg = "sigma_grid_cells must be finite and nonnegative"
        raise ValueError(msg)
    defined = ~np.isnan(values)
    numerator = gaussian_filter(
        np.where(defined, values, 0), sigma_grid_cells, axes=(-2, -1), mode=("reflect", "wrap")
    )
    denominator = gaussian_filter(
        defined.astype(np.float64), sigma_grid_cells, axes=(-2, -1), mode=("reflect", "wrap")
    )
    smooth = np.full_like(values, np.nan)
    np.divide(numerator, denominator, out=smooth, where=defined & (denominator > 0))
    return smooth


def anomaly(field: ArrayLike, sigma_grid_cells: float = 6.0) -> NDArray[np.float64]:
    """Return the residual from the spatially smoothed background."""
    return np.asarray(field, dtype=np.float64) - background(field, sigma_grid_cells)
