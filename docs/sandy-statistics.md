# Sandy signed-rank recomputation

The exact sign-permutation rule was recomputed from the stored matched Sandy
tracks, without rerunning inference. The current 200 km own-low tolerance
export contains 187 baseline, 321 fork and 180 control fixes. The source
track table has SHA-256
`a07bf64f0f19de858800785e00b3c4766f4188c065b44a2585903fa6a377547d`.

Every field of its regenerated `divergence.json`, `tracks.csv`, `tracks.json`,
`tracks.geojson` and `manifest.json` is byte-identical. The divergence SHA-256 is
`68c850fd9b33539d65bd0aea755dbbfef3fb7eb34a773157f45d85bf9a508988`.
The 100 km and 300 km sensitivity tracks and earlier local Sandy exports
also retain every divergence field. No public Sandy number or demo verdict
changes.

Recompute a stored run from the repository root with
`uv run stormfork export <run-dir> --out /tmp/sandy-recomputed`, then run
`uv run stormfork verify /tmp/sandy-recomputed`. Compare its divergence and track
files with the original export; no inference is needed. The source run directory
must retain its `metadata.json`, per-run tracks and field atlases.

## Public summaries: before to after

| Quantity | Before | After |
| --- | ---: | ---: |
| Forecast frames | 41 | 41 |
| Tested steps | 22 | 22 |
| Separated steps at p <= 0.05 | 22 | 22 |
| Peak fork separation (km) | 3230.9896906139243 | 3230.9896906139243 |
| Step 16 fork separation (km) | 1260.5772265388432 | 1260.5772265388432 |
| Step 16 control separation (km) | 32.5301048089169 | 32.5301048089169 |
| Step 16 pairs | 8 | 8 |
| Step 16 p | 0.00390625 | 0.00390625 |
| Last tested step | 21 | 21 |

## Every step: test and stamp inputs

| Step | Pairs before | Pairs after | p before | p after |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 8 | 0.00390625 | 0.00390625 |
| 1 | 8 | 8 | 0.00390625 | 0.00390625 |
| 2 | 8 | 8 | 0.00390625 | 0.00390625 |
| 3 | 8 | 8 | 0.00390625 | 0.00390625 |
| 4 | 8 | 8 | 0.00390625 | 0.00390625 |
| 5 | 7 | 7 | 0.0078125 | 0.0078125 |
| 6 | 8 | 8 | 0.00390625 | 0.00390625 |
| 7 | 8 | 8 | 0.00390625 | 0.00390625 |
| 8 | 7 | 7 | 0.0078125 | 0.0078125 |
| 9 | 8 | 8 | 0.00390625 | 0.00390625 |
| 10 | 8 | 8 | 0.00390625 | 0.00390625 |
| 11 | 8 | 8 | 0.00390625 | 0.00390625 |
| 12 | 7 | 7 | 0.0078125 | 0.0078125 |
| 13 | 8 | 8 | 0.00390625 | 0.00390625 |
| 14 | 8 | 8 | 0.00390625 | 0.00390625 |
| 15 | 8 | 8 | 0.00390625 | 0.00390625 |
| 16 | 8 | 8 | 0.00390625 | 0.00390625 |
| 17 | 8 | 8 | 0.00390625 | 0.00390625 |
| 18 | 8 | 8 | 0.00390625 | 0.00390625 |
| 19 | 8 | 8 | 0.00390625 | 0.00390625 |
| 20 | 6 | 6 | 0.015625 | 0.015625 |
| 21 | 5 | 5 | 0.03125 | 0.03125 |
| 22 | 3 | 3 | None | None |
| 23 | 2 | 2 | None | None |
| 24 | 0 | 0 | None | None |
| 25 | 0 | 0 | None | None |
| 26 | 0 | 0 | None | None |
| 27 | 0 | 0 | None | None |
| 28 | 0 | 0 | None | None |
| 29 | 0 | 0 | None | None |
| 30 | 0 | 0 | None | None |
| 31 | 0 | 0 | None | None |
| 32 | 0 | 0 | None | None |
| 33 | 0 | 0 | None | None |
| 34 | 0 | 0 | None | None |
| 35 | 0 | 0 | None | None |
| 36 | 0 | 0 | None | None |
| 37 | 0 | 0 | None | None |
| 38 | 0 | 0 | None | None |
| 39 | 0 | 0 | None | None |
| 40 | 0 | 0 | None | None |

Null p-values retain the too-few-pairs state. After step 21 the demo retains
the last tested verdict, five confirmed pairs at p=0.03125, with a closed-test label.

The synthetic tie case changes from 0.046875 to 0.0625 for
`[-2, 2, 3, 2, 3, 4]`; those differences are not a Sandy measurement.
