from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, TypeVar

import jax

P = ParamSpec("P")
R = TypeVar("R")

class Transformed(Generic[P, R]):
    apply: Callable[Concatenate[object, jax.Array, P], R]

def transform(f: Callable[P, R], *, apply_rng: bool = True) -> Transformed[P, R]: ...
