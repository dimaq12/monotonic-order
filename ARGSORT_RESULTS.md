# Monotonic-key argsort results

Environment: AMD Ryzen 9 9950X3D, Python 3.12, NumPy 2.2.6, GCC, serial
IdealOrder `0.2.0`. Median of seven warmed calls at `N=1,000,000`.

Command:

```bash
make benchmark-argsort
```

| Key dtype | IdealOrder | NumPy stable argsort | Speedup | Permutation exact |
|---|---:|---:|---:|---:|
| `uint64` | 13.435 ms | 56.414 ms | 4.20x | yes |
| `int64` | 16.167 ms | 56.888 ms | 3.52x | yes |
| `float64` | 17.397 ms | 71.760 ms | 4.12x | yes |

The benchmark measures permutation construction only. Applying the permutation
to a payload is a separate gather whose cost depends on payload shape, dtype
and memory layout. For Python objects, one-time key extraction and list gather
must also be included in end-to-end comparisons.

The current native argsort kernel is serial. Existing OpenMP acceleration for
direct `float64 sort` has not yet been generalized to index scattering, so this
table does not claim a parallel result.
