# Reproducibility

## HDF5 backend decision

I use `h5py==3.16.0` for the official HDF5 sample, pinned in the runtime
dependencies and `uv.lock` on 2026-11-05. The
[PyPI registry](https://pypi.org/pypi/h5py/json) reported 3.16.0 as the latest
stable release, with non-yanked files and a CPython 3.12 macOS arm64 wheel.
This supplies the missing HDF5 backend for the locked h5netcdf 1.8.2 reader;
no model, JAX or data-library pin changed.

## CPU smoke run

Status: passed on 2026-11-05. Runtime: Darwin arm64, CPU, JAX 0.12.0.1,
Python 3.12, on an M5 Mac with 32 GB.

The runner loads Cyclones Mini weights and the official one-step HRES sample,
extracts the six-hour target and forcings with the upstream helper, removes
only the ensemble wrapper, and uses member 0 with `fold_in(PRNGKey(0), 0)`.
The TPU-specific `splash_mha` default is replaced with `triblockdiag_mha`, the
same portable attention implementation used by the official notebook on GPU.
All other model settings and normalization wrappers are retained. A direct
predictor call suffices for this single member and single step; multi-step
rollout and accelerator comparisons remain unmeasured.

```sh
export UV_CACHE_DIR=/private/tmp/stormfork-uv-cache
export UV_PYTHON_INSTALL_DIR=/private/tmp/stormfork-python
uv run python -m stormfork.run.runner \
  --weights data/model/WeatherNextCyclones_Mini_2024.npz \
  --sample data/model/hres_2024-10-07_01.nc
```

Measured result from the final validated implementation:

```json
{"cold_step_s": 10.441342917038128, "lead_hours": 6, "members": 1, "runtime": "cpu", "sizes": {"batch": 1, "lat": 181, "level": 13, "lon": 360, "time": 1}, "variables": 29}
```

The timer starts immediately before the first jitted prediction and ends after
converting every output array to NumPy. It includes cold compilation,
inference and output transfer, and excludes loading weights, reading the
sample and preparing forcings. This is one measured cold step, not a warm
throughput estimate. An earlier process measured 10.752357667079195 seconds.
Both runs completed without OOM and below my chosen 90-second runtime budget.
I can run the real model on this Mac as well as synthetic fixtures.

The runner checks the target variable set and dimensions, finite predictions,
and exact preservation of the input SST missing-value mask. All 29 target
variables passed. Upstream warnings remain visible: single-step gradient
checkpointing is skipped, noise-channel coordinates are inferred, sparse CSR
construction is inefficient, static gather/segment inputs are ignored, and
`cyclone_exists_gaussian_unit_mode` has no normalization scale or location.
This smoke run establishes execution, not forecast skill or cross-device
numerical agreement. CPU peak memory was not measured.

## Small GPU through faster-weathernext

Status: passed on 2026-11-10 at full matmul precision. Eden is a supported
GPU runtime when it runs through faster-weathernext with that precision.

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB, driver 615.78.08,
native Linux, Intel Core i7-10750H, 7.6 GB of RAM. Software: `envs/fwn`
(JAX 0.11.2 with CUDA 12, faster-weathernext 0.2.0 at commit `964be68`,
weathernext at `f2f2c51`), which keeps those pins out of the main lockfile.

`envs/fwn/compare_mini.py` runs one six-hour Cyclones Mini step on the
verified HRES sample through each path, in its own process, with the same
inputs and member key, and compares every field and level with the official
CPU result. The relative RMS difference is the RMS difference divided by the
CPU field's standard deviation; the table gives the largest over all 101.

```sh
(
  cd envs/fwn && uv sync --extra cuda12
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # official GPU path only
  uv run python compare_mini.py run official-cpu --weights W --sample S --out cpu.nc
  uv run python compare_mini.py run fwn-gpu --precision highest --weights W --sample S --out fwn.nc
  uv run python compare_mini.py compare cpu.nc fwn.nc
)
```

| Path | Precision | Max relative RMS vs CPU | JAX peak (GiB) | GPU memory of the process (GiB) | Warm step (s) | Cold step (s) |
|---|---|---:|---:|---:|---:|---:|
| faster-weathernext | highest | 2.30e-5 (vertical velocity, 500 hPa) | 0.87 | 2.27 | 0.76 | 85.4 |
| faster-weathernext | default (TF32) | 1.39e-2 | 0.87 | 2.15 | 0.45 | 32.4 |
| official GPU | highest | out of memory allocating 2.65 GiB | | | | |
| official GPU | default (TF32) | 1.45e-2 | 1.78 | 4.70 | 1.02 | 36.8 |
| official CPU | fp32 | reference | | | 67.6 | 78.1 |

The median relative RMS difference of faster-weathernext at full precision is
3.05e-6. The process figure is what `nvidia-smi` reports for the process after
the step: the allocator's pool plus the CUDA context. The official CPU run on
Eden peaked at 5.11 GB of resident memory.

Decision: the gate is a difference below 1e-4 and peak memory below 5.5 GB.
Only faster-weathernext at full precision passes both. TF32 fails the
difference on both GPU paths, and the official path at full precision does
not fit the card. So on a GPU `stormfork run` patches the model with
faster-weathernext when it is installed, and always computes matrix products
at full precision; `envs/fwn` installs stormfork beside it:

```sh
uv sync --project envs/fwn --extra cuda12
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --project envs/fwn stormfork run <id> --runtime gpu
```

Without `XLA_PYTHON_CLIENT_PREALLOCATE=false`, as in the comparison, JAX
reserves most of the card at start-up, and on a card that also drives a
display the CUDA kernels then fail to load for lack of memory.

### A full experiment on Eden

Two experiments, 8 members, 40 steps, baseline, fork and control, run on
2026-11-10 with the commands above:

| Stage | `milton_south_150km` (s) | `sandy_east_300km` (s) |
|---|---:|---:|
| `ic build`, from the HRES sample and from ERA5 | 5 | 75 |
| `run`, three rollouts of 8 members by 40 steps | 1344 | 1392 |
| `track`, three runs | 170 | 180 |
| `export` | 2 | 2 |

Building the ERA5 input for `sandy_east_300km` on the Mac peaked at 2.7 GB
of resident memory (`/usr/bin/time -l`).

Each run's `forecast.nc` is 2.0 GB. The fork's adjustment ratio is 1.8029
on this GPU; the same fork with one member and one step on the Mac's CPU
gave 1.8029 as well.

That 150 km fork is above the 1.5 publication gate, so the shipped Milton
scenario, `milton_south_75km`, moves the storm 75 km instead; one member and
one step on the CPU give it a ratio of 1.4742. The timings above are for the
150 km runs.

## Verified local assets

Assets remain in ignored `data/model/`. SHA-256 values were recomputed locally
on 2026-11-05 and match the hashes recorded when I downloaded them. Byte counts and base64 MD5
values were verified during that download.

| Filename | Bytes | Base64 MD5 | SHA-256 |
|---|---|---|---|
| WeatherNextCyclones_Mini_2024.npz | 226897594 | R1qGPfvNoqNl62xzJfz6XA== | a1bb151457077248d70a458b1a7b19deacd2926fbcb46eee4ff2691444e3596e |
| hres_2024-10-07_01.nc | 79552583 | KDz7si0OQ8hLxTAQSuyH5Q== | ea28e70fc2b5828af14c9057c45c562a3887da2b1cd951472a841da2630a742d |

Sources:

- [Weights](https://storage.googleapis.com/dm_graphcast/weathernext2/params/WeatherNextCyclones_Mini_%3C2024.npz)
- [HRES sample](https://storage.googleapis.com/dm_graphcast/weathernext2/dataset/source-hres_forecast_init-2024-10-07%2000:00:00_res-1.0_levels-13_steps-01.nc)
- [Official v0.3.0 notebook](https://github.com/google-deepmind/weathernext/blob/v0.3.0/docs/weathernext2/wn2_demo.ipynb)

## Local validation

`UV_CACHE_DIR=/private/tmp/stormfork-uv-cache
UV_PYTHON_INSTALL_DIR=/private/tmp/stormfork-python make check` runs formatting,
lint, strict type checking and the local test suite. Narrow stubs cover only
called APIs; signature tests compare them with the installed pinned sources.

## ERA5 initial-condition comparison

Initialization: 2024-10-07 00:00 UTC. Coarsening: subsample.

RMS differences use both input frames, finite paired cells and all model levels.
The offline header fixture was extracted from the verified local HRES sample above.

Reproduce with `UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run pytest -m network tests/test_era5_layout.py`.

| Variable | RMS difference (native units) |
|---|---:|
| temperature | 0.572889352 |
| geopotential | 35.8886404 |
| u_component_of_wind | 1.28690127 |
| v_component_of_wind | 1.27830581 |
| vertical_velocity | 0.158816221 |
| specific_humidity | 0.000390860296 |
| 2m_temperature | 1.21420613 |
| mean_sea_level_pressure | 63.9120255 |
| 10m_u_component_of_wind | 0.987208225 |
| 10m_v_component_of_wind | 1.22205556 |
| sea_surface_temperature | 1.28076472 |

## Hybrid initial-condition comparison

Initialization: 2024-10-07 00:00 UTC. The hybrid builder reads archived IFS open data (0.25 degree step-0 fields from the 18 UTC and 00 UTC cycles, 13 levels) and takes sea surface temperature from ERA5.

RMS differences to the HRES sample use both input frames, finite paired cells and all
model levels. Normalised differences divide each field's RMS by that field's global
standard deviation in the HRES sample over the same cells.

Reproduce with `UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run pytest -m network tests/test_ifs_open.py`.

### Coarsening check

Hybrid builder, normalised RMS difference by coarsening method:

| Method | vertical_velocity | specific_humidity | mean_sea_level_pressure | Mean |
|---|---:|---:|---:|---:|
| subsample | 0.000242 | 0.000331 | 0.000940 | 0.000504 |
| cell_mean | 0.582211 | 0.047339 | 0.034597 | 0.221382 |

### Source decision with subsample

Only the atmospheric variables enter the rule; surface rows are shown for reference.

| Variable | In rule | ERA5 RMS (native units) | Hybrid RMS (native units) | Ratio |
|---|---:|---:|---:|---:|
| temperature | yes | 0.572889352 | 0.00665909304 | 0.012 |
| geopotential | yes | 35.8886404 | 1.6969539 | 0.047 |
| u_component_of_wind | yes | 1.28690127 | 0.760772163 | 0.591 |
| v_component_of_wind | yes | 1.27830581 | 0.456838915 | 0.357 |
| vertical_velocity | yes | 0.158816221 | 4.57650977e-05 | 0.000 |
| specific_humidity | yes | 0.000390860296 | 1.20118854e-06 | 0.003 |
| 2m_temperature | no | 1.21420613 | 0.00899112124 | 0.007 |
| mean_sea_level_pressure | no | 63.9120255 | 1.16139235 | 0.018 |
| 10m_u_component_of_wind | no | 0.987208225 | 0.00717321785 | 0.007 |
| 10m_v_component_of_wind | no | 1.22205556 | 0.00453767033 | 0.004 |
| sea_surface_temperature | no | 1.28076472 | 1.28076472 | 1.000 |

IC decision: ERA5 is the source for all hindcast inits; u_component_of_wind has a hybrid RMS of 0.591 of ERA5's, limit 0.25.

Coarsening: subsample (mean normalised RMS difference: subsample 0.0005, cell_mean 0.2214); both builders use it.

## Reproducing a published experiment

Open `notebooks/rerun_experiment.ipynb`, pick an `experiment_id` and run every
cell. It builds the initial conditions, checks the edit, rolls out the baseline,
fork and control, tracks the storm, exports the run directory and zips the
published files. Its last cell runs

```sh
stormfork verify <id> --reproduction <run-dir> --by <github-login>
```

which compares the rerun's member and ensemble-mean tracks with the published
ones and prints one JSON entry:

```json
{"by": "login", "runtime": "cpu: device", "date": "YYYY-MM-DD", "code_version": "...", "max_track_diff_km": 0.0, "passed": true}
```

`stormfork rerun <id> --by <login>` runs the same steps for experiments that
start from the HRES sample and compares the result with the published run
through `verify --reproduction`; given a recipe file instead of an id, it only
writes the export files to `--out`.

The commands download what they need on first use: the model weights and the
HRES sample listed under [Verified local assets](#verified-local-assets), and
NCEI's `IBTrACS.ALL.v04r01.nc`. `stormfork fetch` downloads all three at once
and lists their sizes before it starts.

```sh
stormfork fetch
stormfork ic build <id> --init <iso> --out <run-dir>/input.nc
stormfork run <id> --out <run-dir> --members 1 --steps 1
```

Downloads are kept in `$STORMFORK_CACHE_DIR`, else `$XDG_CACHE_HOME/stormfork`,
else `~/.cache/stormfork`. Each one is written to a temporary file and moved
into the cache only after it verifies, so an interrupted download leaves
nothing behind. The weights and the sample must match the SHA-256 in the table
above; `stormfork/downloads.py` holds those pins and records how they were
obtained. NCEI updates the IBTrACS file in place several times a week, so it
has no fixed digest: it is checked against the length the server announces,
and the cached copy is kept until you delete it. With `STORMFORK_OFFLINE=1`
set, every command reads only the cache and names `stormfork fetch` when a
file is missing. `--sample`, `--weights` and `--ibtracs` still take local files
instead of the cache.

`ic build` takes the two input frames from the HRES sample for an
`hres_sample` experiment and from the ERA5 store for the others, with terrain
and land mask from the sample, and lays them out for up to 40 steps. `run`
rolls out the baseline, fork and control of a fork and writes
`<run-dir>/<run>/forecast.nc`, which `stormfork track` reads and which keeps
the surface and cyclone fields and 500 hPa geopotential, member 0's
atlases and previews, and `<run-dir>/metadata.json`, which records the storm's
IBTrACS position at the init. An init before the storm's first IBTrACS record
is valid: the position is then null and the first record's time stands beside
it. Each init of a hindcast is its own experiment, `<hindcast id>-YYYYMMDDHH`,
and records the hindcast's id as `parent_id`. A sweep's null value, such as
0 km, is its fork; every other value is a stop with its own
`<run-dir>/stops/<param>_<value>/forecast.nc` and member 0's pressure atlas,
and `<run-dir>/stops.json` lists each stop with its adjustment ratio and core
land fraction, or the reason the edit check refused it. A stop whose recipe
equals, in content, a published fork already run beside it, with the same
init, seeds, members, steps, model and runtime, copies that fork's forecast
instead of rolling out again. After `track` has written each run's
`tracks.csv`, including each stop's, `stormfork export <run-dir> --out <run-dir>`
completes the directory and packs the manifest and every file it names into
`dist/atlases/<id>.zip`, the asset `scripts/fetch_atlases.py --lock` pins. For a
sweep, `divergence.json` holds every published stop's paired statistics under
`stops`, with the Holm-adjusted p value, and a displacement sweep's manifest
lists the stops for the slider under `sweep`, each with its tracks, divergence
and pressure atlas. A stop the edit check refused or whose adjustment ratio is
above the publication gate is left out.
The zip's entries are sorted, stored uncompressed and dated 1980-01-01, so the
same export gives the same bytes. `rerun` does all of this in one command.

An atlas holds the analysis and up to 40 six-hour steps in its 5 by 9 tiles.
A shorter run fills the first tiles and leaves the rest zero, and the
manifest's `steps` says how many frames are real; `run` and `rerun` refuse
`--steps` above 40.

`run` records in the manifest's `diagnostics` the fork's adjustment ratio, the
land fraction of a relocated core and `bounds_check: passed`, which `verify`
requires of every fork and sweep. [ADR 0014](adr/0014-run-records.md) says
where the ratio is measured for each kind of edit.

### Tolerance

`max_track_diff_km` is the largest great-circle distance between a published and
a rerun position, over every run, step, member and ensemble mean. The entry
passes when it is within the limit for the runtime:

| Rerun runtime | Limit |
|---|---:|
| Same kind as the published run (cpu, gpu or tpu) | 10 km |
| A different kind | 50 km |

These are limits I chose, not measured spreads. Revisit them once reruns on
several runtimes exist. A rerun that lacks a member, step or run that the
published experiment has, or the reverse, is an error rather than a difference.
The wall time of a full rerun has not been recorded yet.

### Reproduction ledger

Append the entry to `site/public/experiments/<id>/reproductions.json` in a pull
request (issue template "reproduction report"). The file is a JSON array of
entries. `scripts/verify_manifest.py` validates every ledger with the manifest's
models. On a pull request, CI also runs it with `--pr-author` and `--base`, and
fails if an added entry's `by`, or a new manifest's `run_by`, is anyone but the
author. The experiment page shows how many people reproduced it, on how many
runtimes.

### Contributor-proposed experiments

The proposer reruns the recipe in the notebook and uploads its zip as a pull
request on the Hugging Face dataset
(`hf upload <owner>/<name>-experiments <zip> --repo-type dataset --create-pr`),
with `run_by` set in the manifest. Once a second person's reproduction has
merged, a maintainer starts the `publish-experiment` workflow with the
experiment id and the Hugging Face pull request revision (`refs/pr/<n>`). It
refuses unless every file matches the merged manifest and a passing
reproduction is by someone other than `run_by`.

### Untrusted archives

Every unzip goes through `stormfork.export.archive.safe_extract`. It refuses,
before writing anything, an absolute path, a drive letter, a backslash, a `..`
component, a symlink, a repeated name, more than 400 entries and a declared
total above 24 MB, the experiment budget in `scripts/perf_budget.py`; the two
caps change only together. It counts the bytes it writes and removes the
destination if they pass the cap. Each PNG must be mode `L` and 1805 x 1629
(atlas), 360 x 181 (preview) or, for `u10.png` and `v10.png` only, 905 x 819
(the 2 degree wind fallback).
