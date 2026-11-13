from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import ParamSpec, TypeVar

from jax._src.basearray import Array as Array

class Device:
    device_kind: str

P = ParamSpec("P")
R = TypeVar("R")

def devices(backend: str | None = None) -> list[Device]: ...
def device_put(x: object) -> Array: ...
def default_device(new_val: Device) -> AbstractContextManager[None]: ...
def jit(fun: Callable[P, R], /) -> Callable[P, R]: ...
