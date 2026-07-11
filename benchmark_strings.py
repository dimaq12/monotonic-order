#!/usr/bin/env python3
"""Variable-length byte ordering: end-to-end and pre-encoded core."""
from __future__ import annotations

import random
import statistics
import time

import numpy as np

from monotonic_order import radix_bytes_argsort, radix_string_argsort


def median_seconds(function, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e9)
    return statistics.median(samples)


def main() -> None:
    size = 200_000
    rng = random.Random(20260719)
    alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789"
    values = [bytes(rng.choice(alphabet) for _ in range(rng.randint(4, 24)))
              for _ in range(size)]
    blob = b"".join(values)
    offsets = np.empty(size+1, dtype=np.uintp)
    offsets[0] = 0
    for index, value in enumerate(values):
        offsets[index+1] = offsets[index]+len(value)

    reference = lambda: sorted(range(size), key=values.__getitem__)
    end_to_end = median_seconds(lambda: radix_string_argsort(values))
    preencoded = median_seconds(lambda: radix_bytes_argsort(blob, offsets))
    python = median_seconds(reference)
    exact = list(radix_bytes_argsort(blob, offsets)) == reference()

    print(f"Variable byte-string argsort, N={size:,}, length=4..24")
    print(f"MonotonicOrder end-to-end: {end_to_end*1e3:.3f} ms")
    print(f"MonotonicOrder pre-encoded: {preencoded*1e3:.3f} ms")
    print(f"Python sorted indices:  {python*1e3:.3f} ms")
    print(f"End-to-end speedup:     {python/end_to_end:.2f}x")
    print(f"Pre-encoded speedup:    {python/preencoded:.2f}x")
    print(f"Exact:                  {exact}")


if __name__ == "__main__":
    main()
