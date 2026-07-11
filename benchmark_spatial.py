#!/usr/bin/env python3
"""Morton 2D encoding and stable permutation benchmark."""
from __future__ import annotations

import statistics
import time

import numpy as np

from ideal_order import morton_argsort, morton_encode, radix_argsort


def median_seconds(function, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e9)
    return statistics.median(samples)


def main() -> None:
    size = 1_000_000
    rng = np.random.default_rng(20260720)
    points = rng.random((size, 2))
    bounds = ((0.0, 1.0), (0.0, 1.0))
    encoded = morton_encode(points, bounds=bounds, bits=32)
    encoding_time = median_seconds(lambda: morton_encode(points, bounds=bounds, bits=32))
    ideal_time = median_seconds(lambda: radix_argsort(encoded.keys))
    numpy_time = median_seconds(lambda: np.argsort(encoded.keys, kind="stable"))
    total_time = median_seconds(lambda: morton_argsort(points, bounds=bounds, bits=32))
    exact = np.array_equal(radix_argsort(encoded.keys),
                           np.argsort(encoded.keys, kind="stable"))
    print(f"Morton 2D pipeline, N={size:,}, 32 bits/axis")
    print(f"Encode only:            {encoding_time*1e3:.3f} ms")
    print(f"IdealOrder key argsort: {ideal_time*1e3:.3f} ms")
    print(f"NumPy key argsort:      {numpy_time*1e3:.3f} ms")
    print(f"Ideal total pipeline:   {total_time*1e3:.3f} ms")
    print(f"Key argsort speedup:    {numpy_time/ideal_time:.2f}x")
    print(f"Pipeline vs encode+np:  {(encoding_time+numpy_time)/total_time:.2f}x")
    print(f"Exact key permutation:  {exact}")


if __name__ == "__main__":
    main()
