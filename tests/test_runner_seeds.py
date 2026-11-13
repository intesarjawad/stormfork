"""A fork with no edit must reproduce the baseline's numbers exactly.

The model weights are not a test fixture, so a stand-in forecast step replaces
the model. It still consumes the member key, which is the part of the runner
that this test exercises.
"""

from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pytest
import xarray as xr

from stormfork.grid import GRID_1DEG
from stormfork.perturb.base import Taper
from stormfork.perturb.relocate import relocate

pytest.importorskip("weathernext")

from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name
from weathernext.weathernext2 import fgn

from stormfork.run import runner

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

zeros_like = cast("Callable[[xr.Dataset], xr.Dataset]", getattr(xr, "zeros_like"))  # noqa: B009

SEED = 7
CONTROL_SEED = SEED + 1000
TAPER = Taper(25.0, 280.0, 400.0)


def example_inputs() -> xr.Dataset:
    lat = GRID_1DEG.lat
    lon = GRID_1DEG.lon
    time = np.array([np.timedelta64(6, "h")], dtype="timedelta64[ns]")
    values = np.full((1, lat.size, lon.size), 280.0)
    return xr.Dataset(
        {"2m_temperature": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def stand_in_predictor(config: fgn.PredictorConfig, ckpt: fgn.CheckPoint) -> runner.PredictorFn:
    """Give every call a value fixed by its key, as the real model does."""
    del config, ckpt

    def predict(
        rng: jax.Array,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        del inputs, forcings
        offset = float(np.asarray(rng).sum())
        return zeros_like(targets_template) + offset

    return predict


@pytest.fixture
def example() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, fgn.PredictorConfig, fgn.CheckPoint]:
    inputs = example_inputs()
    config = get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini")
    ckpt = fgn.CheckPoint(params={}, description="stand-in", license="none")
    return inputs, inputs.copy(deep=True), inputs.copy(deep=True), config, ckpt


def test_a_zero_displacement_fork_is_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
    example: tuple[xr.Dataset, xr.Dataset, xr.Dataset, fgn.PredictorConfig, fgn.CheckPoint],
):
    inputs, targets, forcings, config, ckpt = example
    monkeypatch.setattr(runner, "predictor_fn", stand_in_predictor)
    with jax.default_device(jax.devices("cpu")[0]):
        baseline = runner.rollout(
            inputs, targets, forcings, ckpt, config, num_steps=1, num_members=1, seed=SEED
        )
        fork_inputs = relocate(inputs, TAPER, east_km=0.0, north_km=0.0)
        fork = runner.rollout(
            fork_inputs, targets, forcings, ckpt, config, num_steps=1, num_members=1, seed=SEED
        )
        control = runner.rollout(
            inputs,
            targets,
            forcings,
            ckpt,
            config,
            num_steps=1,
            num_members=1,
            seed=CONTROL_SEED,
        )
    assert fork_inputs.identical(inputs)
    assert set(baseline.sizes) == {"time", "lat", "lon", "sample"}
    assert baseline.sizes["time"] == 1
    assert baseline.sizes["sample"] == 1
    assert baseline.identical(fork)
    assert not baseline.identical(control)


def test_each_step_reaches_the_host_as_numpy_reduced_to_what_the_caller_keeps(
    monkeypatch: pytest.MonkeyPatch,
    example: tuple[xr.Dataset, xr.Dataset, xr.Dataset, fgn.PredictorConfig, fgn.CheckPoint],
):
    inputs, targets, forcings, config, ckpt = example

    def on_device(config: fgn.PredictorConfig, ckpt: fgn.CheckPoint) -> runner.PredictorFn:
        predict = stand_in_predictor(config, ckpt)

        def predict_on_device(
            rng: jax.Array,
            inputs: xr.Dataset,
            targets_template: xr.Dataset,
            forcings: xr.Dataset,
        ) -> xr.Dataset:
            extra = targets_template.assign(extra=targets_template["2m_temperature"])
            predicted = predict(rng, inputs, extra, forcings)
            return predicted.copy(
                data={
                    name: jax.device_put(field.data) for name, field in predicted.data_vars.items()
                }
            )

        return predict_on_device

    kept: list[set[Hashable]] = []

    def keep(chunk: xr.Dataset) -> xr.Dataset:
        kept.append(set(chunk.data_vars))
        return chunk[["2m_temperature"]]

    monkeypatch.setattr(runner, "predictor_fn", on_device)
    with jax.default_device(jax.devices("cpu")[0]):
        forecast = runner.rollout(
            inputs, targets, forcings, ckpt, config, num_steps=1, num_members=2, keep=keep
        )
    assert kept == [{"2m_temperature", "extra"}] * 2
    assert list(forecast.data_vars) == ["2m_temperature"]
    assert isinstance(forecast["2m_temperature"].data, np.ndarray)
