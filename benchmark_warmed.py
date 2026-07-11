#!/usr/bin/env python3
"""Post-warm benchmark: model construction is excluded from every timing."""
from __future__ import annotations

import os
import statistics
import sys
import time

if os.environ.get("IDEAL_ORDER_WARM_BENCH_ENV") != "1":
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "3")
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    env["IDEAL_ORDER_WARM_BENCH_ENV"] = "1"
    os.execve(sys.executable, [sys.executable, __file__], env)

import numpy as np

from ideal_order import IdealOrder


N = 1_000_000
rng = np.random.default_rng(20260711)
reference = rng.standard_normal(N)
incoming = rng.standard_normal(N)

# Everything below is post-warm: neither construction nor first-call library
# initialisation is included in any measured sample.
model = IdealOrder(reference, n_bins=256)
sorted_reference = np.sort(reference)
IdealOrder.sort(incoming)
np.sort(incoming)
_ = model.min, model.max, model.median, model.q1, model.q3, model.mad
_ = model.rank(0.123), model.quantile(0.731)


def median_time_ns(fn, repeats: int) -> tuple[float, object]:
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        result = fn()
        samples.append(time.perf_counter_ns() - t0)
    return float(statistics.median(samples)), result


def fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1e3:.3f} us"
    return f"{ns / 1e6:.3f} ms"


median = float(np.median(reference))

stats = [
    ("min", lambda: model.min, lambda: np.min(reference), 501),
    ("max", lambda: model.max, lambda: np.max(reference), 501),
    ("median", lambda: model.median, lambda: np.median(reference), 101),
    ("q1", lambda: model.q1, lambda: np.quantile(reference, 0.25), 101),
    ("q3", lambda: model.q3, lambda: np.quantile(reference, 0.75), 101),
    ("IQR", lambda: model.iqr,
     lambda: np.quantile(reference, 0.75) - np.quantile(reference, 0.25), 101),
    ("MAD", lambda: model.mad,
     lambda: np.median(np.abs(reference - np.median(reference))), 31),
]

print("idealOrder post-warm benchmark")
print(f"N={N:,}; model and NumPy sorted reference were built before timing")
print("\nStored exact statistics (each NumPy call recomputes from ndarray)")
print(f"{'operation':>10} {'idealOrder':>13} {'NumPy':>13} {'speedup':>12}")
print("-" * 52)
stat_rows = []
for name, ideal_fn, numpy_fn, repeats in stats:
    ideal_ns, ideal_value = median_time_ns(ideal_fn, repeats * 10)
    numpy_ns, numpy_value = median_time_ns(numpy_fn, repeats)
    if not np.isclose(ideal_value, numpy_value, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"{name}: {ideal_value} != {numpy_value}")
    speedup = numpy_ns / ideal_ns
    stat_rows.append((name, ideal_ns, numpy_ns, speedup))
    print(f"{name:>10} {fmt(ideal_ns):>13} {fmt(numpy_ns):>13} {speedup:11.0f}x")

print("\nReference queries")
value = 0.123456
ideal_rank_ns, ideal_rank = median_time_ns(lambda: model.rank(value), 5001)
numpy_rank_ns, numpy_rank_idx = median_time_ns(lambda: np.searchsorted(sorted_reference, value), 5001)
numpy_rank = float(numpy_rank_idx / N)
print(f"scalar rank: ideal={fmt(ideal_rank_ns)}, NumPy sorted-cache={fmt(numpy_rank_ns)}, "
      f"speedup={numpy_rank_ns/ideal_rank_ns:.2f}x, abs_error={abs(ideal_rank-numpy_rank):.6g}")

queries = rng.uniform(model.min, model.max, N)
model.rank_array(queries)
np.searchsorted(sorted_reference, queries)
ideal_bulk_ns, approx = median_time_ns(lambda: model.rank_array(queries), 9)
numpy_bulk_ns, exact_idx = median_time_ns(lambda: np.searchsorted(sorted_reference, queries), 9)
exact = exact_idx / N
print(f"bulk rank 1M: ideal={fmt(ideal_bulk_ns)}, NumPy sorted-cache={fmt(numpy_bulk_ns)}, "
      f"speedup={numpy_bulk_ns/ideal_bulk_ns:.2f}x, max_error={np.max(np.abs(approx-exact)):.6f}")

ideal_q_ns, ideal_q = median_time_ns(lambda: model.quantile(0.731), 5001)
numpy_q_ns, numpy_q = median_time_ns(lambda: np.quantile(reference, 0.731), 101)
print(f"arbitrary quantile: ideal={fmt(ideal_q_ns)}, NumPy raw={fmt(numpy_q_ns)}, "
      f"speedup={numpy_q_ns/ideal_q_ns:.0f}x, abs_error={abs(ideal_q-numpy_q):.6g}")

print("\nRepeated sorting of a new array (both paths already warmed)")
ideal_sort_ns, ideal_sorted = median_time_ns(lambda: model.sort(incoming), 11)
numpy_sort_ns, numpy_sorted = median_time_ns(lambda: np.sort(incoming), 11)
if not np.array_equal(ideal_sorted, numpy_sorted):
    raise AssertionError("sort mismatch")
print(f"sort 1M: ideal={fmt(ideal_sort_ns)}, NumPy={fmt(numpy_sort_ns)}, "
      f"speedup={numpy_sort_ns/ideal_sort_ns:.2f}x, exact=True")

print("\nPersistent memory")
print(f"idealOrder model: {model.storage_bytes:,} bytes")
print(f"NumPy exact sorted-reference cache: {sorted_reference.nbytes:,} bytes")
print(f"reference-cache compression: {sorted_reference.nbytes/model.storage_bytes:.0f}x")
print("NumPy can make stored scalar statistics O(1) by manually caching them; that is the same precompute idea.")
print("idealOrder arbitrary rank/quantile are approximate; NumPy sorted-cache queries are exact.")

model.close()
