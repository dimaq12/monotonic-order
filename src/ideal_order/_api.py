"""Python interface for the compact model and exact float64 order operator."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from enum import Enum
import operator
import unicodedata
import uuid as uuid_module
from typing import Iterable, Union

import numpy as np
from numpy.ctypeslib import ndpointer

from . import _native


_LIB = ct.CDLL(_native.__file__)
_F64_1D = ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
_U64_1D = ndpointer(dtype=np.uint64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
_U8_1D = ndpointer(dtype=np.uint8, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
_UINTP_1D = ndpointer(dtype=np.uintp, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))

_LIB.ideal_order_create.argtypes = [_F64_1D, ct.c_size_t, ct.c_size_t]
_LIB.ideal_order_create.restype = ct.c_void_p
_LIB.ideal_order_destroy.argtypes = [ct.c_void_p]
_LIB.ideal_order_destroy.restype = None

for _name in ("size", "bins", "storage_bytes"):
    _fn = getattr(_LIB, f"ideal_order_{_name}")
    _fn.argtypes = [ct.c_void_p]
    _fn.restype = ct.c_size_t
for _name in ("min", "max", "q1", "median", "q3", "mad"):
    _fn = getattr(_LIB, f"ideal_order_{_name}")
    _fn.argtypes = [ct.c_void_p]
    _fn.restype = ct.c_double

_LIB.ideal_order_rank.argtypes = [ct.c_void_p, ct.c_double]
_LIB.ideal_order_rank.restype = ct.c_double
_LIB.ideal_order_quantile.argtypes = [ct.c_void_p, ct.c_double]
_LIB.ideal_order_quantile.restype = ct.c_double
_LIB.ideal_order_rank_array.argtypes = [ct.c_void_p, _F64_1D, ct.c_size_t, _F64_1D]
_LIB.ideal_order_rank_array.restype = None
_LIB.ideal_order_sort.argtypes = [_F64_1D, ct.c_size_t, _F64_1D, _F64_1D]
_LIB.ideal_order_sort.restype = ct.c_int
_LIB.ideal_order_is_sorted.argtypes = [_F64_1D, ct.c_size_t]
_LIB.ideal_order_is_sorted.restype = ct.c_int
_LIB.ideal_order_unique_sorted.argtypes = [_F64_1D, ct.c_size_t, _F64_1D]
_LIB.ideal_order_unique_sorted.restype = ct.c_size_t
_LIB.ideal_order_argsort_u64.argtypes = [_U64_1D, ct.c_size_t, _UINTP_1D, _UINTP_1D]
_LIB.ideal_order_argsort_u64.restype = ct.c_int
_LIB.ideal_order_lexargsort_u64.argtypes = [
    _U64_1D, ct.c_size_t, ct.c_size_t, _UINTP_1D, _UINTP_1D,
]
_LIB.ideal_order_lexargsort_u64.restype = ct.c_int
_LIB.ideal_order_argsort_bytes.argtypes = [
    _U8_1D, ct.c_size_t, _UINTP_1D, ct.c_size_t, ct.c_int, _UINTP_1D, _UINTP_1D,
]
_LIB.ideal_order_argsort_bytes.restype = ct.c_int
_LIB.ideal_order_hilbert2d_u64.argtypes = [
    _U64_1D, _U64_1D, ct.c_size_t, ct.c_uint, _U64_1D,
]
_LIB.ideal_order_hilbert2d_u64.restype = ct.c_int


ArrayInput = Union[np.ndarray, Iterable[float]]


def _validate_nulls(nulls: str) -> None:
    if nulls not in {"first", "last", "error"}:
        raise ValueError("nulls must be 'first', 'last', or 'error'")


def _null_word(missing: np.ndarray, nulls: str) -> np.ndarray:
    if nulls == "last":
        return np.ascontiguousarray(missing, dtype=np.uint64)
    return np.ascontiguousarray(~missing, dtype=np.uint64)


def _field_words(keys: object, *, descending: bool, nulls: str) -> list[np.ndarray]:
    """Encode one semantic field as most-significant-first uint64 words."""
    _validate_nulls(nulls)
    values = np.asarray(keys)
    if values.ndim != 1:
        raise ValueError("keys must be one-dimensional")

    if values.dtype == np.dtype(np.uint64):
        encoded = np.ascontiguousarray(values)
        return [np.bitwise_not(encoded) if descending else encoded]

    if values.dtype == np.dtype(np.int64):
        bits = np.ascontiguousarray(values).view(np.uint64)
        encoded = np.bitwise_xor(bits, np.uint64(1 << 63))
        return [np.bitwise_not(encoded) if descending else encoded]

    if values.dtype == np.dtype(np.float64):
        contiguous = np.ascontiguousarray(values)
        missing = np.isnan(contiguous)
        if nulls == "error" and np.any(missing):
            raise ValueError("keys contain NaN while nulls='error'")
        bits = contiguous.view(np.uint64)
        sign = (bits & np.uint64(1 << 63)) != 0
        encoded = np.where(sign, np.bitwise_not(bits),
                           np.bitwise_xor(bits, np.uint64(1 << 63))).astype(np.uint64)
        if descending:
            encoded = np.bitwise_not(encoded)
        if np.any(missing):
            encoded[missing] = 0
            return [_null_word(missing, nulls), np.ascontiguousarray(encoded)]
        return [np.ascontiguousarray(encoded)]

    if values.dtype.kind in "Mm" and values.dtype.itemsize == 8:
        contiguous = np.ascontiguousarray(values)
        missing = np.isnat(contiguous)
        if nulls == "error" and np.any(missing):
            raise ValueError("keys contain NaT while nulls='error'")
        ticks = contiguous.view(np.int64)
        encoded = np.bitwise_xor(ticks.view(np.uint64), np.uint64(1 << 63))
        if descending:
            encoded = np.bitwise_not(encoded)
        if np.any(missing):
            encoded[missing] = 0
            return [_null_word(missing, nulls), np.ascontiguousarray(encoded)]
        return [np.ascontiguousarray(encoded)]

    if values.dtype.kind == "O":
        missing = np.zeros(values.size, dtype=bool)
        high = np.zeros(values.size, dtype=np.uint64)
        low = np.zeros(values.size, dtype=np.uint64)
        for index, value in enumerate(values):
            if value is None:
                missing[index] = True
            elif isinstance(value, uuid_module.UUID):
                high[index] = np.uint64(value.int >> 64)
                low[index] = np.uint64(value.int & ((1 << 64)-1))
            else:
                raise TypeError("object keys must be UUID values or None")
        if nulls == "error" and np.any(missing):
            raise ValueError("keys contain None while nulls='error'")
        if descending:
            high = np.bitwise_not(high)
            low = np.bitwise_not(low)
        words = [np.ascontiguousarray(high), np.ascontiguousarray(low)]
        if np.any(missing):
            words.insert(0, _null_word(missing, nulls))
        return words

    raise TypeError("unsupported key dtype; use uint64, int64, float64, datetime64, "
                    "timedelta64, or UUID")


def _native_lexargsort(words: list[np.ndarray]) -> np.ndarray:
    if not words:
        raise ValueError("at least one encoded key word is required")
    size = words[0].size
    if any(word.size != size for word in words):
        raise ValueError("all key fields must have equal length")
    indices = np.empty(size, dtype=np.uintp)
    if size == 0:
        return indices
    workspace = np.empty_like(indices)
    if len(words) == 1:
        key = np.ascontiguousarray(words[0], dtype=np.uint64)
        if not _LIB.ideal_order_argsort_u64(key, size, indices, workspace):
            raise RuntimeError("native radix argsort failed")
        return indices
    matrix = np.ascontiguousarray(np.vstack(words), dtype=np.uint64)
    flat = matrix.ravel()
    if not _LIB.ideal_order_lexargsort_u64(flat, len(words), size, indices, workspace):
        raise RuntimeError("native radix lexargsort failed")
    return indices


def radix_argsort(keys: object, *, descending: bool = False,
                  nulls: str = "last") -> np.ndarray:
    """Return a stable permutation for a supported monotonic key field."""
    values = np.asarray(keys)
    if values.ndim == 1 and values.dtype.kind in "SU":
        return radix_string_argsort(values, descending=descending, nulls=nulls)
    if values.ndim == 1 and values.dtype.kind == "O":
        present = next((value for value in values if value is not None), None)
        if isinstance(present, (str, bytes, np.str_, np.bytes_)):
            return radix_string_argsort(values, descending=descending, nulls=nulls)
    return _native_lexargsort(_field_words(keys, descending=bool(descending), nulls=nulls))


def radix_string_argsort(values: object, *, encoding: str = "utf-8",
                         normalization: str | None = None, casefold: bool = False,
                         descending: bool = False, nulls: str = "last") -> np.ndarray:
    """Stable lexicographic argsort for variable-length bytes or Unicode."""
    _validate_nulls(nulls)
    source = np.asarray(values, dtype=object)
    if source.ndim != 1:
        raise ValueError("string values must be one-dimensional")
    missing_indices: list[int] = []
    original_indices: list[int] = []
    encoded_values: list[bytes] = []
    kind: str | None = None
    for index, value in enumerate(source):
        if value is None:
            missing_indices.append(index)
            continue
        if isinstance(value, (str, np.str_)):
            if kind == "bytes":
                raise TypeError("cannot mix Unicode strings and bytes")
            kind = "str"
            text = str(value)
            if normalization is not None:
                text = unicodedata.normalize(normalization, text)
            if casefold:
                text = text.casefold()
            encoded = text.encode(encoding)
        elif isinstance(value, (bytes, np.bytes_)):
            if kind == "str":
                raise TypeError("cannot mix Unicode strings and bytes")
            if normalization is not None or casefold:
                raise ValueError("normalization and casefold apply only to Unicode strings")
            kind = "bytes"
            encoded = bytes(value)
        else:
            raise TypeError("string keys must contain str, bytes, or None")
        original_indices.append(index)
        encoded_values.append(encoded)
    if missing_indices and nulls == "error":
        raise ValueError("string keys contain None while nulls='error'")

    count = len(encoded_values)
    local = np.empty(count, dtype=np.uintp)
    if count:
        offsets = np.empty(count + 1, dtype=np.uintp)
        offsets[0] = 0
        for index, value in enumerate(encoded_values):
            offsets[index + 1] = offsets[index] + len(value)
        blob = b"".join(encoded_values)
        local = radix_bytes_argsort(blob, offsets, descending=descending)
    ordered_present = np.asarray(original_indices, dtype=np.uintp)[local]
    missing = np.asarray(missing_indices, dtype=np.uintp)
    return (np.concatenate([missing, ordered_present]) if nulls == "first"
            else np.concatenate([ordered_present, missing])).astype(np.uintp, copy=False)


def radix_bytes_argsort(data: object, offsets: object, *,
                        descending: bool = False) -> np.ndarray:
    """Zero-copy-style argsort for a concatenated byte blob and offsets."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(data, dtype=np.uint8)
    else:
        raw = np.ascontiguousarray(data, dtype=np.uint8).ravel()
    boundaries = np.ascontiguousarray(offsets, dtype=np.uintp)
    if boundaries.ndim != 1 or boundaries.size == 0:
        raise ValueError("offsets must be a one-dimensional array of length N+1")
    count = boundaries.size - 1
    indices = np.empty(count, dtype=np.uintp)
    if count == 0:
        if boundaries[0] != 0:
            raise ValueError("empty offsets must contain only zero")
        return indices
    workspace = np.empty_like(indices)
    if not _LIB.ideal_order_argsort_bytes(raw, raw.size, boundaries, count,
                                           int(bool(descending)), indices, workspace):
        raise ValueError("invalid byte blob/offsets or native allocation failure")
    return indices


def enum_keys(values: object, order: dict[Enum, int] | None = None) -> np.ndarray:
    """Materialize explicit signed integer ranks for Enum members."""
    source = list(values)  # type: ignore[arg-type]
    encoded = np.empty(len(source), dtype=np.int64)
    for index, value in enumerate(source):
        if not isinstance(value, Enum):
            raise TypeError("enum_keys expects Enum members")
        rank = order[value] if order is not None else value.value
        if not isinstance(rank, (int, np.integer)):
            raise TypeError("Enum values need integer ranks or an explicit order mapping")
        if rank < np.iinfo(np.int64).min or rank > np.iinfo(np.int64).max:
            raise OverflowError("Enum rank does not fit int64")
        encoded[index] = int(rank)
    return encoded


@dataclass(frozen=True)
class MortonEncoding:
    """Quantized spatial curve keys and an explicit loss report."""

    keys: np.ndarray
    quantized: np.ndarray
    bounds: np.ndarray
    bits: int
    dimensions: int
    clipped_coordinates: int
    curve: str = "morton"
    exact: bool = False


@dataclass(frozen=True)
class HilbertEncoding:
    """Quantized Hilbert 2D keys and an explicit loss report."""

    keys: np.ndarray
    quantized: np.ndarray
    bounds: np.ndarray
    bits: int
    dimensions: int
    clipped_coordinates: int
    curve: str = "hilbert"
    exact: bool = False


def _spread_2d(values: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(values, dtype=np.uint64)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def _spread_3d(values: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(values, dtype=np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_encode(points: object, *, bounds: object, bits: int | None = None,
                  clip: bool = False) -> MortonEncoding:
    """Quantize 2D/3D points and encode their exact Morton cell order."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N,2) or (N,3)")
    dimensions = coordinates.shape[1]
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (dimensions, 2):
        raise ValueError("bounds must have shape (dimensions,2)")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(limits)):
        raise ValueError("points and bounds must be finite")
    if np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("every upper bound must exceed its lower bound")
    maximum_bits = 32 if dimensions == 2 else 21
    selected_bits = maximum_bits if bits is None else operator.index(bits)
    if selected_bits < 1 or selected_bits > maximum_bits:
        raise ValueError(f"bits must lie in [1,{maximum_bits}] for {dimensions}D Morton")

    below = coordinates < limits[:, 0]
    above = coordinates > limits[:, 1]
    outside = below | above
    clipped_coordinates = int(np.count_nonzero(outside))
    if clipped_coordinates and not clip:
        raise ValueError("points lie outside bounds; pass clip=True to clamp explicitly")
    bounded = np.clip(coordinates, limits[:, 0], limits[:, 1]) if clip else coordinates
    levels = (1 << selected_bits)-1
    normalized = (bounded-limits[:, 0])/(limits[:, 1]-limits[:, 0])
    quantized = np.floor(normalized*levels+0.5).astype(np.uint64)

    if dimensions == 2:
        keys = _spread_2d(quantized[:, 0]) | (_spread_2d(quantized[:, 1]) << np.uint64(1))
    else:
        keys = (_spread_3d(quantized[:, 0]) |
                (_spread_3d(quantized[:, 1]) << np.uint64(1)) |
                (_spread_3d(quantized[:, 2]) << np.uint64(2)))
    return MortonEncoding(
        keys=np.ascontiguousarray(keys),
        quantized=np.ascontiguousarray(quantized),
        bounds=limits.copy(),
        bits=selected_bits,
        dimensions=dimensions,
        clipped_coordinates=clipped_coordinates,
    )


def morton_argsort(points: object, *, bounds: object, bits: int | None = None,
                   clip: bool = False, descending: bool = False) -> np.ndarray:
    """Stable argsort by a quantized Morton spatial-curve key."""
    encoded = morton_encode(points, bounds=bounds, bits=bits, clip=clip)
    return radix_argsort(encoded.keys, descending=descending)


def _hilbert_2d_keys(quantized: np.ndarray, bits: int) -> np.ndarray:
    """Native classical xy2d Hilbert state rotation."""
    x = np.ascontiguousarray(quantized[:, 0], dtype=np.uint64)
    y = np.ascontiguousarray(quantized[:, 1], dtype=np.uint64)
    distance = np.empty(len(quantized), dtype=np.uint64)
    if len(quantized) and not _LIB.ideal_order_hilbert2d_u64(
            x, y, len(quantized), bits, distance):
        raise RuntimeError("native Hilbert encoding failed")
    return distance


def hilbert_encode(points: object, *, bounds: object, bits: int = 32,
                   clip: bool = False) -> HilbertEncoding:
    """Quantize 2D points and encode their exact Hilbert cell order."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Hilbert points must have shape (N,2)")
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (2, 2):
        raise ValueError("bounds must have shape (2,2)")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(limits)):
        raise ValueError("points and bounds must be finite")
    if np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("every upper bound must exceed its lower bound")
    selected_bits = operator.index(bits)
    if selected_bits < 1 or selected_bits > 32:
        raise ValueError("bits must lie in [1,32] for Hilbert 2D")
    outside = (coordinates < limits[:, 0]) | (coordinates > limits[:, 1])
    clipped_coordinates = int(np.count_nonzero(outside))
    if clipped_coordinates and not clip:
        raise ValueError("points lie outside bounds; pass clip=True to clamp explicitly")
    bounded = np.clip(coordinates, limits[:, 0], limits[:, 1]) if clip else coordinates
    levels = (1 << selected_bits)-1
    normalized = (bounded-limits[:, 0])/(limits[:, 1]-limits[:, 0])
    quantized = np.floor(normalized*levels+0.5).astype(np.uint64)
    keys = _hilbert_2d_keys(quantized, selected_bits)
    return HilbertEncoding(
        keys=np.ascontiguousarray(keys),
        quantized=np.ascontiguousarray(quantized),
        bounds=limits.copy(),
        bits=selected_bits,
        dimensions=2,
        clipped_coordinates=clipped_coordinates,
    )


def hilbert_argsort(points: object, *, bounds: object, bits: int = 32,
                    clip: bool = False, descending: bool = False) -> np.ndarray:
    """Stable argsort by a quantized Hilbert 2D curve key."""
    encoded = hilbert_encode(points, bounds=bounds, bits=bits, clip=clip)
    return radix_argsort(encoded.keys, descending=descending)


def _field_options(option: object, count: int, name: str) -> list[object]:
    if isinstance(option, (str, bool, np.bool_)):
        return [option] * count
    result = list(option)  # type: ignore[arg-type]
    if len(result) != count:
        raise ValueError(f"{name} must have one entry per key field")
    return result


def radix_lexargsort(*key_fields: object, descending: object = False,
                     nulls: object = "last") -> np.ndarray:
    """Stable lexicographic argsort; fields are most-significant first."""
    if not key_fields:
        raise ValueError("provide at least one key field")
    directions = _field_options(descending, len(key_fields), "descending")
    null_policies = _field_options(nulls, len(key_fields), "nulls")
    words: list[np.ndarray] = []
    for field, reverse, null_policy in zip(key_fields, directions, null_policies):
        if not isinstance(reverse, (bool, np.bool_)):
            raise TypeError("descending entries must be bool")
        if not isinstance(null_policy, str):
            raise TypeError("nulls entries must be strings")
        words.extend(_field_words(field, descending=bool(reverse), nulls=null_policy))
    return _native_lexargsort(words)


def apply_order(payload: object, permutation: object, *, axis: int = 0):
    """Apply a permutation to a NumPy array or a generic Python sequence."""
    order = np.asarray(permutation)
    if order.ndim != 1 or order.dtype.kind not in "iu":
        raise TypeError("permutation must be a one-dimensional integer array")
    if np.any(order < 0):
        raise ValueError("permutation indices must be nonnegative")
    if order.size:
        # Validate before casting: a uint64 greater than INTP_MAX would wrap on
        # conversion. np.bincount requires a signed integer dtype on NumPy 2.0.
        if np.max(order) >= order.size:
            raise ValueError("permutation must contain every index exactly once")
        index = np.ascontiguousarray(order, dtype=np.intp)
        if not np.all(np.bincount(index, minlength=index.size) == 1):
            raise ValueError("permutation must contain every index exactly once")
    else:
        index = np.ascontiguousarray(order, dtype=np.intp)

    if isinstance(payload, np.ndarray):
        normalized_axis = operator.index(axis)
        if normalized_axis < 0:
            normalized_axis += payload.ndim
        if normalized_axis < 0 or normalized_axis >= payload.ndim:
            raise ValueError(f"axis {axis} is out of bounds for dimension {payload.ndim}")
        if payload.shape[normalized_axis] != index.size:
            raise ValueError("payload axis length and permutation size differ")
        return np.take(payload, index, axis=normalized_axis)
    if axis != 0:
        raise ValueError("axis is supported only for NumPy arrays")
    if len(payload) != index.size:  # type: ignore[arg-type]
        raise ValueError("payload length and permutation size differ")
    return [payload[int(i)] for i in index]  # type: ignore[index]


def order_by(payload: object, *, keys: object | None = None, key=None,
             descending: bool = False, nulls: str = "last", axis: int = 0):
    """Stably order arbitrary payloads by a materialized monotonic key."""
    if (keys is None) == (key is None):
        raise ValueError("provide exactly one of keys= or key=")
    if key is not None:
        if axis != 0:
            raise ValueError("key callable is supported only for sequence axis 0")
        materialized = [key(value) for value in payload]  # type: ignore[union-attr]
        if materialized and isinstance(materialized[0], tuple):
            width = len(materialized[0])
            if any(not isinstance(item, tuple) or len(item) != width for item in materialized):
                raise TypeError("all composite keys must be tuples of equal length")
            fields = [np.asarray([item[column] for item in materialized])
                      for column in range(width)]
            permutation = radix_lexargsort(*fields, descending=descending, nulls=nulls)
            return apply_order(payload, permutation, axis=axis)
        keys = np.asarray(materialized)
    permutation = radix_argsort(keys, descending=descending, nulls=nulls)
    return apply_order(payload, permutation, axis=axis)


def _array(values: ArrayInput) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).ravel()


class IdealOrder:
    """Compact reference CDF plus exact operations on new float64 arrays.

    The fitted model retains ``O(n_bins)`` quantile knots, not the training
    array. Stored scalar statistics are exact for the training data. Arbitrary
    fitted ranks and quantiles are approximate. Array ordering operations are
    exact and do not depend on the fitted distribution.
    """

    def __init__(self, data: ArrayInput, n_bins: int = 256):
        x = _array(data)
        if x.size == 0:
            raise ValueError("training data must not be empty")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self._ptr = _LIB.ideal_order_create(x, x.size, n_bins)
        if not self._ptr:
            raise ValueError("training data must contain only finite float64 values")

    def _require_open(self) -> ct.c_void_p:
        ptr = getattr(self, "_ptr", None)
        if not ptr:
            raise RuntimeError("IdealOrder is closed")
        return ptr

    def close(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _LIB.ideal_order_destroy(ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "IdealOrder":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _scalar(self, name: str) -> float:
        return float(getattr(_LIB, f"ideal_order_{name}")(self._require_open()))

    @property
    def n(self) -> int:
        return int(_LIB.ideal_order_size(self._require_open()))

    @property
    def n_bins(self) -> int:
        return int(_LIB.ideal_order_bins(self._require_open()))

    @property
    def storage_bytes(self) -> int:
        return int(_LIB.ideal_order_storage_bytes(self._require_open()))

    min = property(lambda self: self._scalar("min"))
    max = property(lambda self: self._scalar("max"))
    q1 = property(lambda self: self._scalar("q1"))
    median = property(lambda self: self._scalar("median"))
    q3 = property(lambda self: self._scalar("q3"))
    mad = property(lambda self: self._scalar("mad"))
    iqr = property(lambda self: self.q3 - self.q1)

    def rank(self, value: float) -> float:
        """Return an approximate normalized rank in the fitted CDF."""
        return float(_LIB.ideal_order_rank(self._require_open(), float(value)))

    percentile = rank

    def rank_array(self, values: ArrayInput) -> np.ndarray:
        """Return approximate normalized ranks in the fitted CDF."""
        x = _array(values)
        out = np.empty(x.size, dtype=np.float64)
        _LIB.ideal_order_rank_array(self._require_open(), x, x.size, out)
        return out

    def quantile(self, q: float) -> float:
        """Return an approximate fitted quantile; stored knot values are exact."""
        return float(_LIB.ideal_order_quantile(self._require_open(), float(q)))

    def quantile_array(self, qs: ArrayInput) -> np.ndarray:
        q = _array(qs)
        return np.fromiter((self.quantile(v) for v in q), dtype=np.float64, count=q.size)

    @staticmethod
    def sort(values: ArrayInput) -> np.ndarray:
        """Exact stable float64 radix sort with numeric values before NaNs."""
        x = _array(values)
        if x.size == 0:
            return x.copy()
        out = np.empty_like(x)
        workspace = np.empty_like(x)
        if not _LIB.ideal_order_sort(x, x.size, out, workspace):
            raise RuntimeError("exact radix sort failed")
        return out

    @staticmethod
    def sort_reverse(values: ArrayInput) -> np.ndarray:
        out = IdealOrder.sort(values)
        finite = np.count_nonzero(~np.isnan(out))
        out[:finite] = out[:finite][::-1]
        return out

    @staticmethod
    def is_sorted(values: ArrayInput) -> bool:
        x = _array(values)
        return bool(_LIB.ideal_order_is_sorted(x, x.size))

    @staticmethod
    def unique(values: ArrayInput) -> np.ndarray:
        ordered = IdealOrder.sort(values)
        if ordered.size == 0:
            return ordered
        out = np.empty_like(ordered)
        n = int(_LIB.ideal_order_unique_sorted(ordered, ordered.size, out))
        return out[:n].copy()

    @staticmethod
    def bottom_k(values: ArrayInput, k: int) -> np.ndarray:
        ordered = IdealOrder.sort(values)
        return ordered[: max(0, min(int(k), ordered.size))].copy()

    @staticmethod
    def top_k(values: ArrayInput, k: int) -> np.ndarray:
        ordered = IdealOrder.sort(values)
        finite = ordered[: np.count_nonzero(~np.isnan(ordered))]
        count = max(0, min(int(k), finite.size))
        return finite[finite.size-count:].copy() if count else finite[:0].copy()

    @staticmethod
    def exact_quantile(values: ArrayInput, q: float) -> float:
        return float(np.quantile(IdealOrder.sort(values), q))

    @staticmethod
    def count_between(values: ArrayInput, lo: float, hi: float) -> int:
        if lo > hi:
            lo, hi = hi, lo
        ordered = IdealOrder.sort(values)
        left = np.searchsorted(ordered, lo, side="left")
        right = np.searchsorted(ordered, hi, side="right")
        return int(right-left)


sort = IdealOrder.sort
sort_reverse = IdealOrder.sort_reverse
is_sorted = IdealOrder.is_sorted
unique = IdealOrder.unique
bottom_k = IdealOrder.bottom_k
top_k = IdealOrder.top_k
