#!/usr/bin/env python3
"""Reproducible speed, accuracy, and memory benchmark against NumPy."""
from __future__ import annotations

import platform
import statistics
import time
import os
import sys

if os.environ.get("IDEAL_ORDER_BENCH_ENV") != "1":
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "3")
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env["IDEAL_ORDER_BENCH_ENV"] = "1"
    os.execve(sys.executable, [sys.executable, __file__], env)

import numpy as np

from ideal_order import IdealOrder


REPEATS = 9
SIZES = (100_000, 1_000_000, 5_000_000)
rng = np.random.default_rng(20260711)


def elapsed_ms(fn) -> tuple[float, np.ndarray]:
    samples = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples), result


def make_data(kind: str, n: int) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, n)
    if kind == "duplicates":
        return rng.integers(-100, 101, n).astype(np.float64)
    if kind == "reversed":
        return np.arange(n, dtype=np.float64)[::-1].copy()
    raise ValueError(kind)


def equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


print("idealOrder benchmark")
print(f"Python {platform.python_version()} | NumPy {np.__version__} | {platform.processor() or platform.machine()}")
print("Timing includes Python call and output/workspace allocation; median of", REPEATS)
print()
print(f"{'distribution':>13} {'N':>10} {'ideal ms':>11} {'numpy ms':>11} {'speedup':>9} {'exact':>7}")
print("-" * 68)

rows = []
for kind in ("normal", "uniform", "duplicates", "reversed"):
    for n in SIZES:
        x = make_data(kind, n)
        IdealOrder.sort(x)
        np.sort(x, kind="quicksort")
        ideal_ms, ideal_result = elapsed_ms(lambda: IdealOrder.sort(x))
        numpy_ms, numpy_result = elapsed_ms(lambda: np.sort(x, kind="quicksort"))
        exact = equal_with_nan(ideal_result, numpy_result)
        speedup = numpy_ms / ideal_ms
        rows.append((kind, n, ideal_ms, numpy_ms, speedup, exact))
        print(f"{kind:>13} {n:10,d} {ideal_ms:11.3f} {numpy_ms:11.3f} {speedup:9.2f}x {str(exact):>7}")

print("\nCompact model")
training = rng.standard_normal(1_000_000)
t0 = time.perf_counter_ns()
model = IdealOrder(training, n_bins=256)
build_ms = (time.perf_counter_ns() - t0) / 1e6
queries = rng.uniform(model.min, model.max, 100_000)
approx = model.rank_array(queries)
truth = np.searchsorted(np.sort(training), queries, side="left") / training.size
print(f"build N=1,000,000: {build_ms:.3f} ms")
print(f"persistent model: {model.storage_bytes:,} bytes ({training.nbytes / model.storage_bytes:.0f}x smaller than input)")
print(f"rank max error: {np.max(np.abs(approx - truth)):.6f}; theoretical knot interval: {1/model.n_bins:.6f}")

print("\nMemory contract (input excluded)")
for n in SIZES:
    print(f"N={n:>9,d}: ideal output+workspace={16*n/2**20:7.2f} MiB; NumPy output={8*n/2**20:7.2f} MiB")

normal_large = [r for r in rows if r[0] == "normal" and r[1] >= 1_000_000]
geomean = float(np.exp(np.mean(np.log([r[4] for r in normal_large]))))
print(f"\nNormal N>=1M geometric-mean speedup: {geomean:.2f}x")
model.close()
