"""Compare one six-hour Cyclones Mini step across the official and faster-weathernext paths.

faster-weathernext patches the model modules in place, so each path runs in its
own process and writes its prediction; ``compare`` then reads them back:

    python compare_mini.py run official-cpu --weights W --sample S --out cpu.nc
    python compare_mini.py run fwn-gpu --weights W --sample S --out fwn.nc
    python compare_mini.py compare cpu.nc fwn.nc

Every path uses the same inputs, member key and model config; only the
attention kernel and the device differ. The noise is drawn from the key with
jax's counter-based generator, which gives the same numbers on every device.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import haiku as hk
import jax
import numpy as np
import xarray as xr
from weathernext.utils import checkpoint, data_utils
from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name
from weathernext.weathernext2 import fgn

MODEL_CONFIG = "weathernext2/configs/WeatherNextCyclones_Mini"
# The official notebook's GPU kernel; faster-weathernext replaces chunked attention with its own.
ATTENTION = {"official-cpu": "triblockdiag_mha", "official-gpu": "triblockdiag_mha"}
FWN_ATTENTION = "chunked_mha"
GIB = 2**30


def predictor(path: str) -> tuple[fgn.PredictorConfig, object]:
    """Build the single-member predictor for a path, patching the model first for fwn."""
    if path == "fwn-gpu":
        import faster_weathernext  # noqa: PLC0415 - patches the official modules on import use

        faster_weathernext.enable()
    config = get_fiddle_config_by_name(MODEL_CONFIG)
    config.predictor_wrappers = config.predictor_wrappers[:-1]
    transformer = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"]
    transformer.keywords["transformer_kwargs"]["attention_type"] = ATTENTION.get(
        path, FWN_ATTENTION
    )

    @hk.transform
    def forward(inputs: xr.Dataset, targets: xr.Dataset, forcings: xr.Dataset) -> xr.Dataset:
        return fgn.construct_predictor(config)(inputs, targets_template=targets, forcings=forcings)

    return config, jax.jit(forward.apply)


def gpu_process_gib() -> float:
    """GPU memory this process holds, as the driver reports it (pool plus CUDA context)."""
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    used_mib = sum(
        int(used) for pid, used in (row.split(", ") for row in listing.splitlines())
        if int(pid) == os.getpid()
    )  # fmt: skip
    return used_mib / 1024


def run(path: str, weights: Path, sample: Path, out: Path, precision: str) -> dict[str, object]:
    """Run one step on the path, write the prediction and return its timings and memory."""
    config, forward = predictor(path)
    with weights.open("rb") as source:
        ckpt = checkpoint.load(source, fgn.CheckPoint)
    task = config.task
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        xr.load_dataset(sample, engine="h5netcdf"),
        input_variables=task.input_variables,
        target_variables=task.target_variables,
        forcing_variables=task.forcing_variables,
        pressure_levels=task.pressure_levels,
        input_duration=task.input_duration,
        target_lead_times=slice("6h", "6h"),
    )
    device = jax.devices("cpu" if path == "official-cpu" else "gpu")[0]
    rng = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    with jax.default_device(device), jax.default_matmul_precision(precision):
        started = perf_counter()
        forward(ckpt.params, rng, inputs, targets * np.nan, forcings).as_numpy()
        cold_s = perf_counter() - started
        started = perf_counter()
        prediction = forward(ckpt.params, rng, inputs, targets * np.nan, forcings).as_numpy()
        warm_s = perf_counter() - started
    prediction.to_netcdf(out)
    result: dict[str, object] = {
        "path": path,
        "device": device.device_kind,
        "jax": jax.__version__,
        "precision": precision,
        "cold_step_s": cold_s,
        "warm_step_s": warm_s,
        # ru_maxrss is in KiB on Linux.
        "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    if device.platform == "gpu":
        stats = device.memory_stats() or {}
        result["jax_peak_gib"] = stats.get("peak_bytes_in_use", 0) / GIB
        result["gpu_process_gib"] = gpu_process_gib()
    return result


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    """Relative RMS difference of every field and level, scaled by the reference's spread."""
    expected, actual = xr.load_dataset(reference), xr.load_dataset(candidate)
    rows: list[tuple[str, float]] = []
    for name, field in expected.data_vars.items():
        levels = field["level"].values if "level" in field.dims else [None]
        for level in levels:
            x = field if level is None else field.sel(level=level)
            y = actual[name] if level is None else actual[name].sel(level=level)
            a, b = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                msg = f"{name} has a different missing-value mask in {candidate}"
                raise ValueError(msg)
            finite = np.isfinite(a)
            spread = a[finite].std()
            difference = np.sqrt(np.mean((a[finite] - b[finite]) ** 2))
            label = name if level is None else f"{name}@{int(level)}"
            rows.append((label, float(difference / spread) if spread > 0 else 0.0))
    rows.sort(key=lambda row: -row[1])
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "fields": len(rows),
        "max_rel_rms": rows[0][1],
        "median_rel_rms": float(np.median([value for _, value in rows])),
        "worst": rows[:3],
    }


def main() -> None:
    """Run one path or compare two predictions, printing one JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("path", choices=["official-cpu", "official-gpu", "fwn-gpu"])
    one.add_argument("--weights", type=Path, required=True)
    one.add_argument("--sample", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--precision", choices=["highest", "default"], default="highest")
    pair = commands.add_parser("compare")
    pair.add_argument("reference", type=Path)
    pair.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "run":
        result = run(
            arguments.path, arguments.weights, arguments.sample, arguments.out, arguments.precision
        )
    else:
        result = compare(arguments.reference, arguments.candidate)
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
