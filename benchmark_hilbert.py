#!/usr/bin/env python3
"""Hilbert/Morton 2D encoding, ordering and locality benchmark."""
from __future__ import annotations

import statistics
import time

import numpy as np

from ideal_order import hilbert_argsort, hilbert_encode, morton_argsort, morton_encode


def median_seconds(function, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e9)
    return statistics.median(samples)


def locality(points: np.ndarray, order: np.ndarray) -> tuple[float, float]:
    ordered = points[order]
    jumps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    return float(np.mean(jumps)), float(np.quantile(jumps, 0.95))


def main() -> None:
    size = 500_000
    rng = np.random.default_rng(20260722)
    points = rng.random((size, 2))
    bounds = ((0.0, 1.0), (0.0, 1.0))

    # Warm all native paths before collecting medians.
    morton = morton_encode(points, bounds=bounds, bits=32)
    hilbert = hilbert_encode(points, bounds=bounds, bits=32)
    morton_order = morton_argsort(points, bounds=bounds, bits=32)
    hilbert_order = hilbert_argsort(points, bounds=bounds, bits=32)

    morton_encode_time = median_seconds(
        lambda: morton_encode(points, bounds=bounds, bits=32))
    hilbert_encode_time = median_seconds(
        lambda: hilbert_encode(points, bounds=bounds, bits=32))
    morton_total_time = median_seconds(
        lambda: morton_argsort(points, bounds=bounds, bits=32))
    hilbert_total_time = median_seconds(
        lambda: hilbert_argsort(points, bounds=bounds, bits=32))
    morton_mean, morton_p95 = locality(points, morton_order)
    hilbert_mean, hilbert_p95 = locality(points, hilbert_order)

    exact_permutations = (
        np.array_equal(morton_order, np.argsort(morton.keys, kind="stable"))
        and np.array_equal(hilbert_order, np.argsort(hilbert.keys, kind="stable"))
    )

    print(f"Spatial curves, N={size:,}, 32 bits/axis")
    print(f"Morton encode:          {morton_encode_time*1e3:.3f} ms")
    print(f"Hilbert encode:         {hilbert_encode_time*1e3:.3f} ms")
    print(f"Morton total pipeline:  {morton_total_time*1e3:.3f} ms")
    print(f"Hilbert total pipeline: {hilbert_total_time*1e3:.3f} ms")
    print(f"Morton mean jump:       {morton_mean:.9f}")
    print(f"Hilbert mean jump:      {hilbert_mean:.9f}")
    print(f"Mean locality gain:     {morton_mean/hilbert_mean:.3f}x")
    print(f"Morton p95 jump:        {morton_p95:.9f}")
    print(f"Hilbert p95 jump:       {hilbert_p95:.9f}")
    print(f"P95 locality gain:      {morton_p95/hilbert_p95:.3f}x")
    print(f"Exact key permutations: {exact_permutations}")


if __name__ == "__main__":
    main()
