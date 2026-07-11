# Hilbert spatial-key results

Environment: AMD Ryzen 9 9950X3D, Python 3.12, NumPy 2.2.6, GCC, serial
MonotonicOrder `0.6.0`. Median of five warmed calls.

Input: 500,000 uniformly random 2D points in `[0,1]^2`, quantized at 32 bits
per axis. Locality is measured as Euclidean distance between consecutive
original points after sorting by the curve key; smaller is better.

| Measurement | Morton | Hilbert | Ratio |
|---|---:|---:|---:|
| Quantization + encoding | 8.089 ms | 67.691 ms | Hilbert 8.37x slower |
| Total encode + stable argsort | 14.265 ms | 74.416 ms | Hilbert 5.22x slower |
| Mean consecutive-point jump | 0.002053973 | 0.001383655 | Hilbert 1.484x better |
| P95 consecutive-point jump | 0.005846682 | 0.003061956 | Hilbert 1.909x better |

Both MonotonicOrder permutations exactly match NumPy stable argsort of the same
generated keys. Separately, exhaustive grids through 32x32 cells verify every
Hilbert key against an independent scalar `xy2d` reference; traversing a full
grid always moves by one Manhattan cell.

The interpretation is deliberately narrow: Hilbert improves locality on this
distribution, but its state rotations make key construction substantially more
expensive than vectorized Morton bit spreading. Ordering is exact for quantized
cells, while conversion from continuous coordinates remains lossy. Neither
curve guarantees that every nearby point becomes adjacent.

Reproduce with:

```bash
make benchmark-hilbert
```
