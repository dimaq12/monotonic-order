# Changelog

All notable changes are documented here. The project follows semantic versioning
while its public API is in alpha development.

## 0.6.0 — 2026-07-11

- add native Hilbert 2D encoding for one to 32 bits per axis;
- verify Hilbert keys against an independent exhaustive reference;
- report locality and construction-cost tradeoffs against Morton;
- fix permutation validation for NumPy 2.0 on Python 3.9;
- run the complete 36-test suite across the GitHub Actions matrix.

## 0.5.0 — 2026-07-11

- add explicit-bounds Morton 2D/3D spatial encoding;
- report quantized cells, clipping count and lossy-coordinate semantics;
- add spatial pipeline benchmarks.

## 0.4.0 — 2026-07-11

- add native variable-length byte radix ordering;
- add Unicode normalization and case-folding policy;
- add integer-ranked Enum key materialization.

## 0.3.0 — 2026-07-11

- add native multiword lexicographic argsort;
- add UUID, datetime and timedelta codecs;
- add stable multi-field ordering and tuple-key object ordering.

## 0.2.0 — 2026-07-11

- add stable radix argsort for `uint64`, `int64` and `float64` monotonic keys;
- add arbitrary-payload permutation helpers;
- package the native core through CPython's stable `abi3` interface.

## 0.1.0 — 2026-07-11

- introduce the compact reference-distribution model;
- add exact float64 sorting and stored statistics;
- document the mathematical operator and approximation boundary.
