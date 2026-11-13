from dataclasses import dataclass
from typing import Protocol, TypedDict

import xarray as xr

from weathernext.utils.fiddle_config_io import TaskConfig

class MeshKeywords(TypedDict):
    transformer_kwargs: dict[str, object]

class MeshConstructor(Protocol):
    keywords: MeshKeywords

class NoisyFunctionKwargs(TypedDict):
    mesh_model_ctor: MeshConstructor

class PredictorKwargs(TypedDict):
    noisy_function_kwargs: NoisyFunctionKwargs

@dataclass
class PredictorConfig:
    task: TaskConfig
    predictor_constructor: object
    predictor_kwargs: PredictorKwargs
    predictor_wrappers: list[dict[str, object]]

@dataclass(frozen=True)
class CheckPoint:
    params: object
    description: str
    license: str

class Predictor(Protocol):
    def __call__(
        self, inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset
    ) -> xr.Dataset: ...

def construct_predictor(config: PredictorConfig) -> Predictor: ...
