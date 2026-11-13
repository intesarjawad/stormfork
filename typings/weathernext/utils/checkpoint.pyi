from typing import BinaryIO, TypeVar

T = TypeVar("T")

def load(source: BinaryIO, typ: type[T]) -> T: ...
