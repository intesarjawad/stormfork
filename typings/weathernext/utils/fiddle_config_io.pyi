from typing import Protocol

from weathernext.weathernext2.fgn import PredictorConfig

class TaskConfig(Protocol):
    input_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    forcing_variables: tuple[str, ...]
    pressure_levels: tuple[int, ...]
    input_duration: str

def get_fiddle_config_by_name(name: str) -> PredictorConfig: ...
