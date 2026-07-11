# Measured results

Environment: AMD Ryzen 9 9950X3D, Python 3.12.3, NumPy 2.2.6, GCC 13.3,
three nearby OpenMP cores. Median of nine calls, including Python call and
output/workspace allocation. Command: `make benchmark`.

| Distribution | N | MonotonicOrder | NumPy | Speedup | Exact |
|---|---:|---:|---:|---:|---:|
| normal | 100,000 | 0.783 ms | 0.274 ms | 0.35x | yes |
| normal | 1,000,000 | 2.243 ms | 3.876 ms | 1.73x | yes |
| normal | 5,000,000 | 15.492 ms | 25.151 ms | 1.62x | yes |
| uniform | 1,000,000 | 2.296 ms | 3.836 ms | 1.67x | yes |
| uniform | 5,000,000 | 15.882 ms | 25.537 ms | 1.61x | yes |
| bounded integer duplicates | 1,000,000 | 1.556 ms | 1.679 ms | 1.08x | yes |
| reversed | 1,000,000 | 0.813 ms | 3.731 ms | 4.59x | yes |

The normal-distribution geometric-mean speedup for `N >= 1M` was **1.68x**.
NumPy remained faster at `N=100K`; this implementation does not claim a
universal speed advantage for small arrays.

For a one-million-element training array with `K=256`:

- construction: 173.828 ms;
- persistent model: 2,128 bytes, 3,759x smaller than the 8 MB input;
- measured maximum normalized-rank error: 0.002328;
- knot interval bound: 1/256 = 0.003906.

Out-of-place sort working memory, excluding input:

| N | MonotonicOrder output + workspace | NumPy output |
|---:|---:|---:|
| 100,000 | 1.53 MiB | 0.76 MiB |
| 1,000,000 | 15.26 MiB | 7.63 MiB |
| 5,000,000 | 76.29 MiB | 38.15 MiB |

The speedup is purchased with one additional `8*N` radix workspace. The
persistent learned model itself remains `O(K)`.
