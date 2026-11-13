# 0012: Stabilize the perturbation API

Status: accepted.

## Context

Atmospheric edits and balance diagnostics are useful independently of a forecast
model. Callers need a small import surface, explicit units and a grid contract
that supports different resolutions without changing the edit mathematics.

## Decision

The planned v0.1.0 public API is `stormfork.perturb`, exporting exactly
`Taper`, `relocate`, `scale_intensity`, `sst_anomaly`, `noise`, `validate`,
`adjustment_ratio` and `PhysicalBoundsError` through `__all__`. Breaking changes
require a major version increment. Computation uses numpy and scipy on xarray
datasets without network or filesystem access.

Fields use named `lat` and `lon` dimensions, with optional frame and level axes
as documented by each operation. Coordinates are ascending and equally spaced,
with matching latitude and longitude spacing in degrees. Longitudes cover the
global interval [0, 360) from zero without a duplicate endpoint. Latitude may
cover a subset of the globe. `Grid.from_dataset` reads spacing from coordinates;
the API does not promise support for regional longitude grids, descending axes,
unequal spacing or curvilinear grids. Default WeatherNext variable names can be
overridden with a `names` mapping. Unknown keys raise `ValueError`; explicitly
mapped variables must exist or raise `KeyError`. Editable fields must use
floating-point storage, including for identity edits.

Taper centres use degrees and radii use km. Relocation distances and SST anomaly
widths use km; intensity factors and noise amplitudes are dimensionless. Pressure
uses Pa, temperatures K, specific humidity kg/kg and pressure levels hPa.

`adjustment_ratio(fork, baseline, taper, *, names=None)` accepts datasets with at
least two entries along `time`. Callers supply the initial state at index 0 and
the six-hour forecast at index 1; later steps are ignored. MSLP has `time`, `lat`
and `lon` dimensions in any order, with optional `member` selected at index 0
and a singleton `batch` dimension squeezed away. Time coordinates, when present,
must have a six-hour first interval. The spatial coordinates must match between
datasets.

The diagnostic divides fork RMS pressure change by baseline RMS pressure change
over cell centres whose great-circle distance lies between half the taper radius
and the full radius, including both boundaries. RMS gives each cell equal weight.
Only the selected annulus must have finite tendencies. A zero baseline tendency,
empty annulus, invalid pressure layout or nonfinite annulus tendency raises
`DiagnosticError`, available from `stormfork.perturb.diagnostics` without extending
the package's `__all__`. Grid contract violations raise `ValueError`.

## Evidence and limits

Synthetic tests check the diagnostic against explicit RMS values, annulus
boundaries, longitude wrapping, member selection and custom variable names.
The public relocation test constructs a single-variable Gaussian pressure low
at both 1° and 0.25° spacing and checks its displacement within one grid cell.

A ratio near 1 means similar pressure-change magnitude; a larger ratio flags
excess change for investigation, without establishing its cause. Synthetic tests establish arithmetic,
not a measured distribution of atmospheric adjustment or an empirical publication
threshold. [Perturbation limits](../perturbations.md) describe the balance
limitation and chosen experimental cutoffs. Manifest recording belongs with forecast execution and export, which
are not implemented here.
