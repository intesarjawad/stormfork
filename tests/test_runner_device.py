import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

pytest.importorskip("weathernext")

from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name
from weathernext.weathernext2 import fgn

from stormfork.run import runner


def mini_config() -> fgn.PredictorConfig:
    return get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini")


def attention(config: fgn.PredictorConfig) -> object:
    transformer = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"]
    return transformer.keywords["transformer_kwargs"]["attention_type"]


def installed(monkeypatch: pytest.MonkeyPatch, *, backend: str, faster: bool) -> list[str]:
    """Pretend jax runs on backend, with or without faster-weathernext; record its enable calls."""
    enabled: list[str] = []
    module = ModuleType(runner.FASTER_WEATHERNEXT)
    setattr(module, "enable", lambda: enabled.append(backend))  # noqa: B010
    monkeypatch.setitem(sys.modules, runner.FASTER_WEATHERNEXT, module)
    monkeypatch.setattr(runner, "default_backend", lambda: backend)

    def find_spec(name: str) -> ModuleSpec | None:
        return ModuleSpec(name, None) if faster else None

    monkeypatch.setattr(runner, "find_spec", find_spec)
    return enabled


def test_a_gpu_with_faster_weathernext_patches_the_model_before_building_it(
    monkeypatch: pytest.MonkeyPatch,
):
    enabled = installed(monkeypatch, backend="gpu", faster=True)
    config = mini_config()
    original = attention(config)
    inference = runner.inference_config(config)
    assert enabled == ["gpu"]
    assert attention(inference) == "chunked_mha"
    assert attention(config) == original
    assert len(inference.predictor_wrappers) == len(config.predictor_wrappers) - 1


@pytest.mark.parametrize(("backend", "faster"), [("gpu", False), ("cpu", True), ("cpu", False)])
def test_other_cpus_and_gpus_run_the_official_portable_kernel(
    monkeypatch: pytest.MonkeyPatch, backend: str, *, faster: bool
):
    enabled = installed(monkeypatch, backend=backend, faster=faster)
    assert attention(runner.inference_config(mini_config())) == "triblockdiag_mha"
    assert enabled == []
