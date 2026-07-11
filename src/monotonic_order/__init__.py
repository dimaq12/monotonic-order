"""Public Python API for MonotonicOrder."""

from ._api import (
    MonotonicOrder,
    HilbertEncoding,
    MortonEncoding,
    apply_order,
    bottom_k,
    enum_keys,
    hilbert_argsort,
    hilbert_encode,
    is_sorted,
    morton_argsort,
    morton_encode,
    order_by,
    radix_argsort,
    radix_bytes_argsort,
    radix_lexargsort,
    radix_string_argsort,
    sort,
    sort_reverse,
    top_k,
    unique,
)

__version__ = "0.7.0"

# Transitional source compatibility. New code should use MonotonicOrder.
IdealOrder = MonotonicOrder

__all__ = [
    "MonotonicOrder",
    "IdealOrder",
    "HilbertEncoding",
    "MortonEncoding",
    "apply_order",
    "bottom_k",
    "enum_keys",
    "hilbert_argsort",
    "hilbert_encode",
    "is_sorted",
    "morton_argsort",
    "morton_encode",
    "order_by",
    "radix_argsort",
    "radix_bytes_argsort",
    "radix_lexargsort",
    "radix_string_argsort",
    "sort",
    "sort_reverse",
    "top_k",
    "unique",
    "__version__",
]
