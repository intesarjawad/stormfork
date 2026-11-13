import argparse
import copy
import json
import sys
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

import haiku as hk
import jax
import numpy as np
import xarray as xr
from jax.random import PRNGKey, fold_in
from weathernext.utils import checkpoint, data_utils
from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name
from weathernext.weathernext2 import fgn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# The upstream rollout module has no type stubs; this narrow view covers the one call used.
rollout_utils = import_module("weathernext.utils.rollout")
generate_chunks = cast(
    "Callable[..., Iterator[xr.Dataset]]",
    getattr(rollout_utils, "chunked_prediction_generator_multiple_runs"),  # noqa: B009
)
default_backend = cast("Callable[[], str]", getattr(jax, "default_backend"))  # noqa: B009
default_matmul_precision = cast(
    "Callable[[str], AbstractContextManager[None]]",
    getattr(jax, "default_matmul_precision"),  # noqa: B009
)

FASTER_WEATHERNEXT = "faster_weathernext"
GPU_PRECISION = "highest"

# Block sizes tuned for a single v5e TPU in the model's demo notebook.
TPU_BLOCK_SIZES = {
    "block_q": 128,
    "block_kv": 128,
    "block_kv_compute": 128,
    "block_q_dkv": 128,
    "block_kv_dkv": 128,
    "block_kv_dkv_compute": 128,
}


class PredictorFn(Protocol):
    """One jitted forecast step, called by the upstream rollout loop."""

    def __call__(
        self,
        rng: jax.Array,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        """Return the target variables for one lead time."""
        ...


def device_kind() -> str:
    """Name the device jax computes on, as its runtime reports it."""
    return jax.devices()[0].device_kind


def uses_faster_weathernext() -> bool:
    """Whether the model runs through faster-weathernext: on a GPU, where envs/fwn installs it."""
    return default_backend() == "gpu" and find_spec(FASTER_WEATHERNEXT) is not None


def inference_config(config: fgn.PredictorConfig) -> fgn.PredictorConfig:
    """Copy the predictor config without its ensemble wrapper, suited to this device.

    The default Splash kernel needs a TPU. CPUs and GPUs use the portable
    triblockdiag kernel; a TPU keeps Splash with block sizes for one v5e. On a
    GPU with faster-weathernext installed, the model modules are patched first
    and chunked attention selected, which it replaces with its fused kernel:
    the same function in a fraction of the memory a small card lacks.
    """
    inference = copy.deepcopy(config)
    inference.predictor_wrappers = inference.predictor_wrappers[:-1]
    transformer = inference.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"]
    transformer_kwargs = transformer.keywords["transformer_kwargs"]
    backend = default_backend()
    if uses_faster_weathernext():
        enable = cast("Callable[[], object]", getattr(import_module(FASTER_WEATHERNEXT), "enable"))  # noqa: B009
        enable()
        transformer_kwargs["attention_type"] = "chunked_mha"
    elif backend in ("cpu", "gpu"):
        transformer_kwargs["attention_type"] = "triblockdiag_mha"
    elif backend == "tpu":
        transformer_kwargs.update(TPU_BLOCK_SIZES)
    return inference


def predictor_fn(config: fgn.PredictorConfig, ckpt: fgn.CheckPoint) -> PredictorFn:
    """Build one jitted forecast step from a checkpoint and a single-member config."""

    @hk.transform
    def forward(
        inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset
    ) -> xr.Dataset:
        predictor = fgn.construct_predictor(config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def predict(
        rng: jax.Array,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        return forward.apply(ckpt.params, rng, inputs, targets_template, forcings)

    return jax.jit(predict)


def on_host(dataset: xr.Dataset) -> xr.Dataset:
    """Copy a prediction's device arrays to NumPy, which the tracker's indexing needs."""
    return dataset.copy(
        data={name: np.asarray(field.data) for name, field in dataset.data_vars.items()}
    )


def rollout(  # noqa: PLR0913, PLR0917
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    ckpt: fgn.CheckPoint,
    config: fgn.PredictorConfig,
    num_steps: int = 40,
    num_members: int = 8,
    seed: int = 0,
    *,
    keep: "Callable[[xr.Dataset], xr.Dataset]" = lambda chunk: chunk,
) -> xr.Dataset:
    """Roll out an ensemble for ``num_steps`` six-hour steps, one chunk per step.

    Member ``i`` folds ``i`` into ``seed``, so the first members stay the same
    whatever the ensemble size. A fork with no edit and the same seed is the
    baseline bit for bit, because the model sees identical inputs and keys.
    Each chunk moves to host memory as NumPy arrays as soon as it is predicted,
    reduced by ``keep`` to the fields the caller needs, so neither the device
    nor the host holds the whole forecast of every variable and level.

    Raises:
        ValueError: A count is not positive, or the templates are shorter than
            the requested number of steps.
    """
    if num_steps < 1 or num_members < 1:
        msg = "num_steps and num_members must be at least 1"
        raise ValueError(msg)
    for name, dataset in (("targets_template", targets_template), ("forcings", forcings)):
        available = dataset.sizes.get("time", 0)
        if available < num_steps:
            msg = f"{name} has {available} steps, fewer than the {num_steps} requested"
            raise ValueError(msg)
    rngs = np.stack([fold_in(PRNGKey(seed), member) for member in range(num_members)])
    # GPUs default to TF32 matrix products, which put one step about 1e-2
    # relative RMS from the CPU result; full precision keeps it within 1e-4.
    precision = (
        default_matmul_precision(GPU_PRECISION) if default_backend() == "gpu" else nullcontext()
    )
    with precision:
        # The one-device loop tags each chunk with a scalar sample; promote it so
        # combine_by_coords can order members and lead times into one hypercube.
        chunks = [
            on_host(keep(chunk)).expand_dims("sample")
            for chunk in generate_chunks(
                predictor_fn=predictor_fn(inference_config(config), ckpt),
                rngs=rngs,
                inputs=inputs,
                targets_template=targets_template.isel(time=slice(0, num_steps)) * np.nan,
                forcings=forcings.isel(time=slice(0, num_steps)),
                num_steps_per_chunk=1,
                num_samples=num_members,
            )
        ]
    combine_by_coords = cast(
        "Callable[[list[xr.Dataset]], object]",
        getattr(xr, "combine_by_coords"),  # noqa: B009
    )
    predictions = combine_by_coords(chunks)
    if not isinstance(predictions, xr.Dataset):
        msg = "ensemble rollout did not produce a dataset"
        raise TypeError(msg)
    return predictions


def cpu_smoke(weights_path: Path, sample_path: Path) -> tuple[xr.Dataset, float]:
    """Run one Mini member for six hours, timing cold inference and transfer."""
    config = inference_config(
        get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini")
    )
    with weights_path.open("rb") as source:
        ckpt = checkpoint.load(source, fgn.CheckPoint)
    sample = xr.load_dataset(sample_path, engine="h5netcdf")
    task = config.task
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        sample,
        input_variables=task.input_variables,
        target_variables=task.target_variables,
        forcing_variables=task.forcing_variables,
        pressure_levels=task.pressure_levels,
        input_duration=task.input_duration,
        target_lead_times=slice("6h", "6h"),
    )

    @hk.transform
    def forward(i: xr.Dataset, t: xr.Dataset, f: xr.Dataset) -> xr.Dataset:
        predictor = fgn.construct_predictor(config)
        return predictor(i, targets_template=t, forcings=f)

    with jax.default_device(jax.devices("cpu")[0]):
        rng = fold_in(PRNGKey(0), 0)
        predict = jax.jit(forward.apply)
        started = perf_counter()
        predictions = predict(ckpt.params, rng, inputs, targets * np.nan, forcings)
        # NumPy conversion waits for all device work, including asynchronous dispatch.
        predictions = predictions.as_numpy()
        elapsed_s = perf_counter() - started
    if predictions.sizes != targets.sizes or set(predictions) != set(targets):
        msg = "CPU prediction does not match the target layout"
        raise ValueError(msg)
    for name, array in predictions.data_vars.items():
        # The official NaNCleaner restores the input SST land mask.
        values = array.values
        if name == "sea_surface_temperature":
            expected_mask = inputs[name].isel(time=-1).isnull().values
            if not np.array_equal(np.isnan(values[:, 0]), expected_mask):
                msg = "CPU prediction changed the SST missing-value mask"
                raise ValueError(msg)
            values = np.where(np.isnan(values), 0, values)
        if not bool(np.isfinite(values).all()):
            msg = f"CPU prediction contains non-finite values: {name}"
            raise ValueError(msg)
    return predictions, elapsed_s


def main() -> None:
    """Measure the local CPU smoke run with verified, predownloaded assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    predictions, elapsed_s = cpu_smoke(args.weights, args.sample)
    result = {
        "runtime": "cpu",
        "cold_step_s": elapsed_s,
        "members": 1,
        "lead_hours": 6,
        "variables": len(predictions.data_vars),
        "sizes": dict(predictions.sizes),
    }
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
