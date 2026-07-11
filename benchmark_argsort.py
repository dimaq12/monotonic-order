#!/usr/bin/env python3
"""Stable monotonic-key argsort benchmark against NumPy."""
from __future__ import annotations

import statistics
import time

import numpy as np

from monotonic_order import radix_argsort


def median_seconds(function, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e9)
    return statistics.median(samples)


def main() -> None:
    size = 1_000_000
    rng = np.random.default_rng(20260717)
    arrays = {
        "uint64": rng.integers(0, 2**63, size=size, dtype=np.uint64),
        "int64": rng.integers(-(2**62), 2**62, size=size, dtype=np.int64),
        "float64": rng.standard_normal(size),
    }
    print(f"Stable argsort benchmark, N={size:,}")
    print(f"{'dtype':>10} {'MonotonicOrder':>12} {'NumPy':>12} {'speedup':>10} {'exact':>7}")
    for name, values in arrays.items():
        radix_argsort(values)
        np.argsort(values, kind="stable")
        ideal = median_seconds(lambda: radix_argsort(values))
        numpy = median_seconds(lambda: np.argsort(values, kind="stable"))
        permutation = radix_argsort(values)
        expected = np.argsort(values, kind="stable")
        exact = np.array_equal(permutation, expected)
        print(f"{name:>10} {ideal*1e3:>9.3f} ms {numpy*1e3:>9.3f} ms "
              f"{numpy/ideal:>9.2f}x {str(exact):>7}")


if __name__ == "__main__":
    main()
