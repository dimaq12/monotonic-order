# Variable-length string ordering results

Environment: AMD Ryzen 9 9950X3D, Python 3.12, NumPy 2.2.6, GCC, serial
IdealOrder `0.4.0`. Median of five warmed calls.

Input: 200,000 random byte strings, lengths 4 through 24.

| Path | Time | Relative to Python |
|---|---:|---:|
| IdealOrder end-to-end Python bytes | 68.125 ms | 0.60x |
| IdealOrder pre-encoded blob + offsets | 13.671 ms | 2.99x |
| Python stable `sorted` indices | 40.902 ms | reference |

Both IdealOrder paths produce the exact same stable permutation as Python.

The result exposes the real boundary: the native variable-byte MSD radix core
is faster once data already uses an Arrow/database-style byte blob and offsets,
but materializing that representation from a Python `list[bytes]` dominates and
makes the current end-to-end convenience path slower than Python sorting.

Therefore `radix_bytes_argsort(blob, offsets)` is the performance API.
`radix_string_argsort(list_of_strings)` is primarily a correctness and
convenience API until key materialization is further optimized.
