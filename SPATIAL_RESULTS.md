# Morton spatial-key results

Environment: AMD Ryzen 9 9950X3D, Python 3.12, NumPy 2.2.6, GCC, serial
IdealOrder `0.5.0`. Median of five warmed calls.

Input: one million random 2D points in `[0,1]^2`, quantized at 32 bits per
axis into one 64-bit Morton key.

| Stage | Time |
|---|---:|
| Morton quantization + encoding | 20.142 ms |
| IdealOrder stable key argsort | 12.729 ms |
| NumPy stable key argsort | 57.106 ms |
| IdealOrder total encode+argsort | 32.655 ms |

Key permutation speedup over NumPy: **4.49x**. Total spatial pipeline speedup
against the same Morton encoding followed by NumPy stable argsort: **2.37x**.

The resulting order is exact for the quantized Morton cells and stable for
collisions. It is not an exact representation of continuous coordinates:
quantization is lossy, and Morton order is a chosen space-filling-curve order,
not lexicographic coordinate order or a guarantee that every geometric neighbor
is adjacent.
