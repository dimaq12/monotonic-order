# Multiword lexargsort results

Environment: AMD Ryzen 9 9950X3D, Python 3.12, NumPy 2.2.6, GCC, serial
MonotonicOrder `0.3.0`. Median of seven warmed calls at `N=1,000,000`.

Command:

```bash
make benchmark-lexargsort
```

Input: four independent `int64` fields. Fields are passed to both methods in
the same most-significant-to-least-significant semantic order.

| Method | Time | Exact permutation |
|---|---:|---:|
| MonotonicOrder `radix_lexargsort` | 46.696 ms | yes |
| NumPy `lexsort` | 247.781 ms | reference |

Measured speedup: **5.31x**.

This benchmark measures permutation construction only. It excludes payload
gather and any Python object field extraction. Runtime grows linearly with the
number of encoded 64-bit words; UUID uses two words, while a nullable field may
add a separate null-order word.
