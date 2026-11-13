# Atlas sizes

Synthetic full-size analytic fields on a 181 x 360 grid, 41 frames. Wind atlases use 1° sampling; all previews are 360 x 181. These measurements do not estimate model-output compression. Real-run measurements are pending.

Sizes are filesystem bytes; MB means 1,000,000 bytes. PNGs use lossless
8-bit grayscale with Pillow optimize=True. Each run contains 41 frames.

| Run | File | Atlas bytes | Preview bytes |
| --- | --- | ---: | ---: |
| baseline | mslp | 205404 | 6311 |
| baseline | u10 | 180778 | 336 |
| baseline | v10 | 171042 | 336 |
| baseline | t2m | 128826 | 4739 |
| baseline | z500 | 270071 | 8026 |
| baseline | tp6h | 187710 | 143 |
| baseline | Total | 1143831 | 19891 |

Run including previews: 1163722 bytes.

## Perf budget

Gzip level 9, sizes in bytes; KB means 1,000 bytes.
The bundle is every .js and .mjs file in the build output.

Commands, run from the repository root:

```
(cd site && pnpm run build)
uv run python scripts/perf_budget.py --bundle site/dist --record docs/benchmarks.md
```

| File | Gzip bytes |
| --- | ---: |
| assets/index-DpiF1g5O.js | 241746 |
| assets/relief-MVc-NPZ7.js | 8214 |
| Total | 249960 |

Cap: 250000 bytes.
