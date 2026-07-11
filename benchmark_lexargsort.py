#!/usr/bin/env python3
"""Multi-column stable lexargsort benchmark against NumPy."""
from __future__ import annotations

import statistics
import time

import numpy as np

from ideal_order import radix_lexargsort


def median_seconds(function, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e9)
    return statistics.median(samples)


def main() -> None:
    size = 1_000_000
    rng = np.random.default_rng(20260718)
    fields = tuple(rng.integers(-(2**31), 2**31, size=size, dtype=np.int64)
                   for _ in range(4))
    radix_lexargsort(*fields)
    np.lexsort(tuple(reversed(fields)))
    ideal = median_seconds(lambda: radix_lexargsort(*fields))
    numpy = median_seconds(lambda: np.lexsort(tuple(reversed(fields))))
    actual = radix_lexargsort(*fields)
    expected = np.lexsort(tuple(reversed(fields)))
    print(f"Stable 4-field lexargsort, N={size:,}")
    print(f"IdealOrder: {ideal*1e3:.3f} ms")
    print(f"NumPy:      {numpy*1e3:.3f} ms")
    print(f"Speedup:    {numpy/ideal:.2f}x")
    print(f"Exact:      {np.array_equal(actual, expected)}")


if __name__ == "__main__":
    main()
