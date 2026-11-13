from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike

from stormfork.export.quantise import (
    BYTE_MAX,
    FILE_NAMES,
    decode,
    decoding_interval,
    field_for_file,
    field_metadata,
    quantise,
)


@pytest.mark.parametrize("file", FILE_NAMES)
def test_full_size_round_trip(file: str):
    field = field_for_file(file)
    values = np.random.default_rng(100).uniform(field.lower, field.upper, (41, 181, 360))
    codes = quantise(values, file)
    decoded = decode(codes, file)
    lower, upper = decoding_interval(codes, file)
    assert codes.dtype == np.uint8
    assert np.all(values >= lower - 1e-10)
    assert np.all(values <= upper + 1e-10)
    if field.encoding == "linear":
        assert np.max(abs(values - decoded)) <= (field.upper - field.lower) / 510 + 1e-10
    else:
        assert np.max(abs(np.sqrt(values) - np.sqrt(decoded))) <= np.sqrt(field.upper) / 510
        bound = np.maximum(decoded - lower, upper - decoded)
        assert np.all(abs(values - decoded) <= bound + 1e-15)


@pytest.mark.parametrize("file", FILE_NAMES)
def test_endpoints_clipping_and_all_byte_codes(file: str):
    field = field_for_file(file)
    np.testing.assert_array_equal(
        quantise([field.lower - 1, field.lower, field.upper, field.upper + 1], file),
        [0, 0, 255, 255],
    )
    codes = np.arange(BYTE_MAX + 1, dtype=np.uint8)
    np.testing.assert_array_equal(quantise(decode(codes, file), file), codes)
    lower, upper = decoding_interval(codes, file)
    np.testing.assert_allclose(upper[:-1], lower[1:], rtol=0, atol=0)
    assert lower[0] == pytest.approx(field.lower)
    assert upper[-1] == pytest.approx(field.upper)


def test_square_root_bound_is_tight_in_native_units():
    codes = np.arange(256, dtype=np.uint8)
    lower, upper = decoding_interval(codes, "tp6h")
    decoded = decode(codes, "tp6h")
    np.testing.assert_allclose(lower, 0.1 * (np.maximum(codes.astype(float) - 0.5, 0) / 255) ** 2)
    np.testing.assert_allclose(upper, 0.1 * (np.minimum(codes.astype(float) + 0.5, 255) / 255) ** 2)
    assert np.max(np.maximum(decoded - lower, upper - decoded)) == pytest.approx(
        0.1 * (1 / 255 - 1 / (4 * 255**2))
    )


@given(st.sampled_from(FILE_NAMES), st.floats(min_value=-1e6, max_value=1e6))
def test_clipped_input_belongs_to_decoding_interval(file: str, value: float):
    field = field_for_file(file)
    lower, upper = decoding_interval(quantise(value, file), file)
    clipped = np.clip(value, field.lower, field.upper)
    assert lower - 1e-10 <= clipped <= upper + 1e-10


def test_wind_ties_to_even_and_browser_speed():
    assert quantise(-40, "v10") == 42
    assert quantise(30, "u10") == 191
    assert np.hypot(decode(191, "u10"), decode(42, "v10")) == pytest.approx(50.118, abs=0.001)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("file", FILE_NAMES)
def test_nonfinite_values_rejected(file: str, value: float):
    with pytest.raises(ValueError, match="finite"):
        quantise([value], file)


@pytest.mark.parametrize("value", [-1, 256, 0.5, np.inf, np.nan])
@pytest.mark.parametrize("operation", [decode, decoding_interval])
def test_invalid_codes_rejected(value: float, operation: Callable[[ArrayLike, str], object]):
    with pytest.raises(ValueError, match="integers"):
        operation([value], "mslp")


def test_field_metadata_and_unknown_files():
    entries = field_metadata(wind_resolution_deg=2)
    assert len(entries) == 5
    assert entries[1] == {
        "name": "wind10",
        "files": ["u10", "v10"],
        "range": [-60, 60],
        "encoding": "linear",
        "resolution_deg": 2,
    }
    assert all(entry["resolution_deg"] == 1 for entry in entries if entry["name"] != "wind10")
    entries[0]["files"].append("wrong")
    assert field_metadata()[0]["files"] == ["mslp"]
    with pytest.raises(ValueError, match="unknown atlas"):
        quantise([0], "wind10")
    with pytest.raises(ValueError, match="1 or 2"):
        field_metadata(wind_resolution_deg=3)
