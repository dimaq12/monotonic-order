"""Python interface for the compact model and exact float64 order operator."""
from __future__ import annotations

import ctypes as ct
from typing import Iterable

import numpy as np
from numpy.ctypeslib import ndpointer

from . import _native


_LIB = ct.CDLL(_native.__file__)
_F64_1D = ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))

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


ArrayInput = np.ndarray | Iterable[float]


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
