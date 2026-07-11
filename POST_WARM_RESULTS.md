# Post-warm benchmark

AMD Ryzen 9 9950X3D, Python 3.12.3, NumPy 2.2.6, `N=1,000,000`.
The `IdealOrder` model and NumPy sorted-reference cache were constructed before
timing. Median timings; run with `make benchmark-warmed`.

## Stored exact statistics

Each NumPy call below recomputes the statistic from its unchanged ndarray.

| Operation | idealOrder | NumPy | Speedup |
|---|---:|---:|---:|
| min | 391 ns | 64.870 us | 166x |
| max | 321 ns | 72.214 us | 225x |
| median | 330 ns | 5.579 ms | 16,907x |
| Q1 | 320 ns | 8.340 ms | 26,064x |
| Q3 | 320 ns | 9.405 ms | 29,391x |
| IQR | 591 ns | 17.742 ms | 30,021x |
| MAD | 321 ns | 13.108 ms | 40,834x |

NumPy can also make these reads `O(1)` if the application manually computes
and stores the scalar results. `IdealOrder` packages that precomputation into
the model and preserves its immutable-data contract.

## Reference queries and new arrays

| Operation | idealOrder | NumPy | Speedup | Contract |
|---|---:|---:|---:|---|
| scalar rank | 281 ns | 692 ns | 2.46x | approximate vs exact sorted cache |
| 1M ranks | 20.698 ms | 105.078 ms | 5.08x | max normalized error 0.002373 |
| arbitrary quantile | 281 ns | 9.413 ms | 33,497x | approximate vs raw exact NumPy |
| sort new 1M array | 2.189 ms | 3.830 ms | 1.75x | both exact and warmed |

Persistent reference storage:

- `IdealOrder`: 2,128 bytes;
- exact NumPy sorted-reference copy: 8,000,000 bytes;
- compression: 3,759x.

The comparison intentionally distinguishes stored exact scalar statistics
from approximate arbitrary rank/quantile queries. Exact arbitrary reference
queries require retaining substantially more information.
