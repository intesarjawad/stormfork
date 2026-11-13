# 0002: Choose the initial-condition source by RMS difference to HRES

Status: accepted.

## Context

The model's published inputs are HRES analyses on a one-degree grid with 13 levels. Hindcast initializations need the same fields from an archive we can read. ERA5 reaches every date but is a different analysis system. ECMWF's open data archive carries the IFS analysis at 0.25 degrees, the same analysis the HRES sample comes from, so it should sit closer to the sample once reduced to one degree. That reduction is not documented for the sample, so the build also tests how it was made.

## Decision

Build 2024-10-07 00 UTC two ways and compare each with the shipped HRES sample:

- ERA5: the existing builder on the ARCO store.
- Hybrid: archived IFS open data for the six atmospheric variables and the four surface fields the open data carry, with ERA5 sea surface temperature. Step-0 fields of the 18 UTC and 00 UTC cycles supply the two input frames; geopotential is the open-data height times 9.80665.

The comparison uses both input frames, finite paired cells and all model levels. The hybrid is the source for all hindcast initializations from 2023-07-12 onward if, for every atmospheric input variable, its RMS difference is at most 25 percent of the ERA5 builder's. Otherwise ERA5 is the source.

The one-degree reduction is chosen first: build the hybrid with `subsample` (every fourth point) and with `cell_mean` (weighted cell means), and keep the method with the lower mean normalised RMS difference over `vertical_velocity`, `specific_humidity` and `mean_sea_level_pressure`. Each field's RMS is divided by that field's global standard deviation in the HRES sample, so pascals do not outweigh kilograms per kilogram. Both builders then use that method for the decision.

The tables are in `docs/reproducibility.md`. The network test `tests/test_ifs_open.py` rewrites them and the outcome below.

## Limits

The decision rests on one initialization. On 2026-11-07 the Google bucket `ecmwf-open-data` returned 404 for the 0.25-degree step-0 index of 2023-07-12 and 2024-02-27 and 200 for 2024-02-29, and the AWS bucket answered with throttling errors. Initializations before the first available 0.25-degree file cannot use the hybrid builder as written, whatever the outcome below says.

## Outcome

IC decision: ERA5 is the source for all hindcast inits; u_component_of_wind has a hybrid RMS of 0.591 of ERA5's, limit 0.25.

Coarsening: subsample (mean normalised RMS difference: subsample 0.0005, cell_mean 0.2214); both builders use it.
