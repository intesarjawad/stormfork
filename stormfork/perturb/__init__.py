"""Pure xarray edits and balance diagnostics, with numpy and scipy computation.

Datasets use named lat/lon axes in any order, with optional frame and level axes.
Coordinates must ascend at equal, matching spacing in degrees; longitude must
cover [0, 360) globally from zero without a duplicate endpoint. Latitude may span
a subset of the globe. Spacing is read from the coordinates, including at 0.25°.
Optional names mappings translate supported WeatherNext names to existing data
variables. Unknown keys raise ValueError; missing mapped variables raise KeyError.
Editable fields must have floating-point storage, including for identity edits.

Taper defines a circular weight on (lat, lon), with a centre in degrees and radius
in km. relocate translates atmospheric anomalies by east/north distances in km on
each frame/level slice, retaining native field units. scale_intensity multiplies
those anomalies by a dimensionless factor on the same layout. sst_anomaly adds
kelvin on lat/lon slices, using a centre in degrees, width in km, and levels in
hPa. noise adds seeded, dimensionless-amplitude perturbations scaled to each
slice's spatial standard deviation in native units.

validate checks matching datasets with pressure in Pa, temperature in K and
specific humidity in kg/kg. PhysicalBoundsError reports bounds or preservation
failures on these arrays. adjustment_ratio returns a dimensionless MSLP tendency
RMS ratio from pressure in Pa on time/lat/lon axes and optional member; indices
0 and 1 represent the initial and six-hour states, using member index 0 and
squeezing singleton batch axes. Time coordinates must have a six-hour interval.

SST edits and noise do not rebalance geopotential or other coupled fields.
The first forecast step adjusts them; the ratio monitors pressure changes
without a calibrated balance threshold. See docs/perturbations.md for chosen
experimental limits, their rationale and data provenance.

These exports form the planned v0.1.0 public API. Breaking changes require
a major version increment.
"""

from stormfork.perturb.base import Taper
from stormfork.perturb.diagnostics import adjustment_ratio
from stormfork.perturb.noise import noise
from stormfork.perturb.relocate import relocate
from stormfork.perturb.scale_intensity import scale_intensity
from stormfork.perturb.sst_anomaly import sst_anomaly
from stormfork.perturb.validate import PhysicalBoundsError, validate

__all__ = [
    "PhysicalBoundsError",
    "Taper",
    "adjustment_ratio",
    "noise",
    "relocate",
    "scale_intensity",
    "sst_anomaly",
    "validate",
]
