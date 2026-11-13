# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

No release has been tagged. The Python package and site use development version 0.0.0.

### Added

- Python CLI for reproducible forecast edits, ensemble tracking, verification and atlas export.
- Whole-run WN2 collector with ensemble mean, member 0 and unit checks.
- Local model input-contract and stub-signature checks.
- Experiment list in `experiments/*.yaml` and `stormfork scenarios list`.
- Rerun notebook, `stormfork verify --reproduction`, the reproduction ledger and its pull request author check.
- `safe_extract` for untrusted archives and the `publish-experiment` workflow.
- Experiment titles and limits, the method, perturbation and recipe guides, `scripts/figures.py` and the DOI and recipe-schema checks.
- Displacement chart of landfall latitude for sweeps, and the null-fork check that `stormfork verify` makes on a sweep.
- `stormfork ic build` and `stormfork run` build the model input and roll out the baseline, fork and control of a fork from local weights and the HRES sample; `stormfork rerun` rolls out and tracks them.
- `stormfork run` also writes member 0's atlases and `metadata.json`, so `stormfork rerun` completes through the export; runs shorter than 40 steps fill the first atlas tiles.
- `stormfork fetch` and a download cache: `ic build`, `run` and `rerun` fetch the pinned weights, HRES sample and IBTrACS on first use and verify them; `STORMFORK_OFFLINE=1` reads only the cache.
- Manifests credit ECMWF for the HRES data every initial-condition source reads.
- `stormfork run` rolls out a sweep: its null value as the fork, and every other value as a stop with its forecast and pressure atlas, listed in `stops.json` with its diagnostics or the reason the edit check refused it.
- `stormfork run` records each fork's adjustment ratio, the land fraction of a relocated core and the passed bounds check; `stormfork verify` requires the bounds check of every fork and sweep.
- `stormfork export` also writes `dist/atlases/<id>.zip`, byte for byte the same for the same export, for the atlases release.

### Changed

- A storm's track followed through extratropical transition takes its pressure from the low itself rather than the tropical-cyclone field, carries sub-grid positions, rejects a deeper neighbour entering the search area, keeps the search radius after sub-grid refinement, wraps wind interpolation at the longitude seam, and treats a missing minimum as missing rather than infinitely shallow.
- Manifest schema version 2: the storm's position at the init is null for an init before its first IBTrACS record, which `run` no longer refuses, and the first record's time is recorded beside it.
- Each hindcast init is its own experiment, `<hindcast id>-YYYYMMDDHH`, with the hindcast's id as `parent_id`.
- Edits without a taper have their adjustment ratio measured around the storm's initial position; a relocation takes the larger of its start and destination.
- A sweep's values must be distinct and include the null fork.
- The site's front page is a ledger of the fork's twenty-four storms beside a 1-bit relief of the run's real sea-level pressure, with each member's track burnt into it; the whole site uses two inks.

### Fixed

- Pages requires checksum-verified published atlas archives before deployment.
- Station reports are aligned within thirty minutes of forecast times, including window boundaries.
- Signed-rank tests use exact sign permutations of the observed midranks, including ties and zero differences.
- Scheduled backfill restores archived chunks before selecting the next interval.
- Scoreboard comparators use matching stations, initializations and leads, with equal sample counts and physical units.
- Benchmark commands preserve the repository working directory, and the GPU recipe uses its environment from the root.
- IBTrACS times read from NCEI's NetCDF are rounded to the second, so the storm's position at the init is found.
- The edit check accepts undefined sea surface temperature in cells with land, and the edits clip specific humidity at zero, so the three HRES-sample forks pass on the real sample.
