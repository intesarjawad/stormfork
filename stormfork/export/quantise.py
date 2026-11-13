from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

BYTE_MAX = 255
Encoding = Literal["linear", "sqrt"]


class FieldMetadata(TypedDict):
    """Browser decoding contract, with ranges in each field's native units."""

    name: str
    files: list[str]
    range: list[float]
    encoding: Encoding
    resolution_deg: int


@dataclass(frozen=True)
class Field:
    """Fixed physical range shared by a field's single-channel files."""

    name: str
    files: tuple[str, ...]
    lower: float
    upper: float
    encoding: Encoding = "linear"


FIELDS = (
    Field("mslp", ("mslp",), 92000, 105000),  # Pa
    Field("wind10", ("u10", "v10"), -60, 60),  # m/s per component
    Field("t2m", ("t2m",), 220, 320),  # K
    Field("z500", ("z500",), 48000, 59000),  # m²/s²
    Field("tp6h", ("tp6h",), 0, 0.1, "sqrt"),  # m per six hours
)
FILE_NAMES = tuple(file for field in FIELDS for file in field.files)


def field_for_file(file: str) -> Field:
    """Find the physical decoding contract for one PNG basename.

    Raises:
        ValueError: The basename is not one of the six exported files.
    """
    for field in FIELDS:
        if file in field.files:
            return field
    msg = f"unknown atlas file {file!r}; expected one of {FILE_NAMES}"
    raise ValueError(msg)


def field_metadata(*, wind_resolution_deg: int = 1) -> list[FieldMetadata]:
    """Describe five fields, with the optional 2° wind sampling explicitly marked.

    Raises:
        ValueError: Wind resolution is neither 1 nor 2 degrees.
    """
    if wind_resolution_deg not in (1, 2):
        msg = "wind_resolution_deg must be 1 or 2"
        raise ValueError(msg)
    return [
        FieldMetadata(
            name=field.name,
            files=list(field.files),
            range=[field.lower, field.upper],
            encoding=field.encoding,
            resolution_deg=wind_resolution_deg if field.name == "wind10" else 1,
        )
        for field in FIELDS
    ]


def quantise(values: ArrayLike, file: str) -> NDArray[np.uint8]:
    """Encode native units to bytes, clipping before ties-to-even rounding.

    The round-trip guarantee applies to clipped values. Linear error is at most
    (upper - lower) / 510, apart from floating-point roundoff. For precipitation,
    the half-step bound is sqrt(upper) / 510 in square-root space. A constant
    half-step in metres is incompatible with square-root encoding; use
    decoding_interval for the exact, asymmetric interval in native units.

    Raises:
        ValueError: Values are nonfinite or the file is unknown.
    """
    field = field_for_file(file)
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{file}: required finite values before quantisation"
        raise ValueError(msg)
    clipped = np.clip(array, field.lower, field.upper)
    normalised = (
        np.sqrt(clipped) / np.sqrt(field.upper)
        if field.encoding == "sqrt"
        else (clipped - field.lower) / (field.upper - field.lower)
    )
    return np.asarray(np.rint(normalised * BYTE_MAX), dtype=np.uint8)


def _checked_codes(codes: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(codes, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(
        (array < 0) | (array > BYTE_MAX) | (array != np.floor(array))
    ):
        msg = "codes must be integers in [0, 255]"
        raise ValueError(msg)
    return array


def _decode_normalised(values: NDArray[np.float64], field: Field) -> NDArray[np.float64]:
    return np.asarray(
        (values * np.sqrt(field.upper)) ** 2
        if field.encoding == "sqrt"
        else field.lower + values * (field.upper - field.lower),
        dtype=np.float64,
    )


def decode(codes: ArrayLike, file: str) -> NDArray[np.float64]:
    """Decode integer bytes to native units using the browser's arithmetic.

    Raises:
        ValueError: Codes are not bytes or the file is unknown.
    """
    return _decode_normalised(_checked_codes(codes) / BYTE_MAX, field_for_file(file))


def decoding_interval(
    codes: ArrayLike, file: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return tight closed bounds on clipped input values for each byte.

    Half-integer boundaries include both ties for an error bound; ties-to-even
    determines which byte owns the boundary itself. Squaring these boundaries
    gives an exact native-unit bound for precipitation, including codes 0/255.

    Raises:
        ValueError: Codes are not bytes or the file is unknown.
    """
    array = _checked_codes(codes)
    field = field_for_file(file)
    return (
        _decode_normalised(np.maximum(array - 0.5, 0) / BYTE_MAX, field),
        _decode_normalised(np.minimum(array + 0.5, BYTE_MAX) / BYTE_MAX, field),
    )
