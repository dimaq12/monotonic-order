# IdealOrder

[![tests](https://github.com/dimaq12/order/actions/workflows/test.yml/badge.svg)](https://github.com/dimaq12/order/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://github.com/dimaq12/order/blob/main/pyproject.toml)
[![CPython stable ABI](https://img.shields.io/badge/CPython-abi3-4B8BBE)](https://docs.python.org/3/c-api/stable.html)
[![release](https://img.shields.io/github/v/release/dimaq12/order?display_name=tag)](https://github.com/dimaq12/order/releases)

**Stable radix ordering for monotonic keys, plus a compact `O(K)` reference
distribution model.** IdealOrder returns reusable permutations instead of
moving arbitrary payload objects inside the native kernel.

## Why IdealOrder

- exact stable argsort for `uint64`, `int64`, IEEE-754 `float64`, datetime,
  timedelta and UUID keys;
- lexicographic multi-field ordering with per-field direction and missing-value
  policy;
- variable-length bytes and Unicode ordering without Python comparators in the
  radix passes;
- explicit Morton 2D/3D and Hilbert 2D spatial-curve keys;
- compact quantile reference state with stored min/max/median/quartiles/MAD;
- C11 core and CPython `abi3` extension targeting Python 3.9+.

| Read next | Contents |
|---|---|
| [Theory](THEORY.md) | Mathematical contracts, exactness boundaries and proofs |
| [Monotonic-key epic](EPIC_MONOTONIC_KEYS.md) | Supported domains and remaining roadmap |
| [Benchmarks](ARGSORT_RESULTS.md) | Reproducible performance measurements |
| [Changelog](CHANGELOG.md) | Release history |
| [Contributing](CONTRIBUTING.md) | Development and verification workflow |

The training array is sorted only in temporary construction memory; it is never
retained by the finished model.

The complete mathematical specification, proofs of the radix ordering
contract, approximation boundary and warmed-object semantics are in
[THEORY.md](THEORY.md).

The monotonic-key roadmap, shipped phases and remaining arbitrary-payload work
are tracked in [EPIC_MONOTONIC_KEYS.md](EPIC_MONOTONIC_KEYS.md).

## Mathematical contract

The compressed reference operator is

```text
P_K(mu_N) = (q_0, ..., q_K; min, q1, median, q3, max, MAD),
```

where `q_j = F_N^-1(j/K)`. It provides monotone approximate arbitrary ranks
and quantiles. For strictly increasing knots the probability-cell resolution
is `1/K`; repeated values create intrinsically wider rank intervals. Values at
stored quantile knots and the stored scalar statistics are exact.

Exact sorting is a separate universal operator. A finite non-NaN IEEE-754
value is mapped to an unsigned key `tau(x)` such that

```text
x < y  iff  tau(x) < tau(y).
```

A stable five-pass radix factorisation (13+13+13+13+12 bits) sorts those keys.
The inverse map returns the original values. NaN-containing arrays use the
payload-preserving path and place all NaNs last. Therefore correctness does
not depend on how closely new data matches the training distribution.
Monotone arrays and bounded integer-valued float domains have exact linear
fast paths before radix execution.

Exact arbitrary rank/contains queries against discarded training data are
intentionally not claimed: those require `Omega(N)` information in the worst
case. `rank()` and `quantile()` are named as compressed-reference estimates;
all array operations (`sort`, `unique`, `top_k`, `bottom_k`, `count_between`,
`is_sorted`) are exact.

## Install

From the latest tagged release:

```bash
python -m pip install "idealorder @ git+https://github.com/dimaq12/order.git@v0.6.0"
```

For local development:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The distribution name is `idealorder`; the import package is `ideal_order`.
Installation compiles the bundled C11 core as a platform extension. Portable
builds deliberately avoid `-march=native`. To opt into OpenMP when the compiler
and runtime support it:

```bash
IDEAL_ORDER_OPENMP=1 python -m pip install .
```

The native loader uses CPython's stable ABI (`abi3`) targeting Python 3.9+, so
one wheel can serve multiple CPython versions on the same platform.

## Use

```python
import numpy as np
from ideal_order import IdealOrder, sort

training = np.random.default_rng(1).normal(size=1_000_000)
with IdealOrder(training, n_bins=256) as model:
    print(model.storage_bytes, model.median, model.rank(0.0))

ordered = sort(np.random.default_rng(2).normal(size=1_000_000))
```

Order arbitrary payloads by one monotonic numeric key per item:

```python
import numpy as np
from ideal_order import order_by, radix_argsort

records = [{"name": "c", "score": 2},
           {"name": "a", "score": 1},
           {"name": "b", "score": 1}]

ordered = order_by(records, key=lambda row: np.int64(row["score"]))
# a, b, c -- equal score=1 records remain stable

keys = np.array([30, -5, 10], dtype=np.int64)
permutation = radix_argsort(keys)
```

`radix_argsort` supports exact `uint64`, `int64`, `float64`, NumPy
`datetime64/timedelta64`, and UUID keys. `radix_lexargsort` composes multiple
fields with per-field direction and missing-value policy.

Version 0.4 adds exact variable-length bytes and Unicode ordering, explicit
normalization/casefold policy, and integer-ranked Enum helpers. For data already
stored as a concatenated byte blob plus offsets, use `radix_bytes_argsort` to
avoid Python key-materialization overhead. See [STRING_RESULTS.md](STRING_RESULTS.md).

Version 0.5 adds explicit-bounds Morton 2D/3D encoding. The returned
`MortonEncoding` includes quantized cells, clipping count and `exact=False` so
curve ordering cannot be confused with lossless coordinate ordering. See
[SPATIAL_RESULTS.md](SPATIAL_RESULTS.md).

Version 0.6 adds independently verified Hilbert 2D encoding with the same
explicit quantization and clipping contract:

```python
from ideal_order import hilbert_argsort, hilbert_encode

encoded = hilbert_encode(points, bounds=((0, 1), (0, 1)), bits=32)
permutation = hilbert_argsort(points, bounds=((0, 1), (0, 1)), bits=32)
```

Hilbert improves consecutive-point locality in the recorded random-cloud
benchmark, while costing more to encode than Morton. See
[HILBERT_RESULTS.md](HILBERT_RESULTS.md).
Measured permutation-only results are recorded in
[ARGSORT_RESULTS.md](ARGSORT_RESULTS.md). Multi-field results are in
[LEXARGSORT_RESULTS.md](LEXARGSORT_RESULTS.md).

## Native development build

```bash
cd idealOrder
make
make test
make asan
make benchmark
make benchmark-warmed
make benchmark-hilbert
```

For reproducible parallel performance, set `OMP_NUM_THREADS=3`,
`OMP_PROC_BIND=close`, and `OMP_PLACES=cores` in the process environment.
`make benchmark` does this automatically. Tune these values for a different
CPU topology.

## Memory

- Persistent model: `sizeof(model) + 8*(K+1)` bytes, about 2 KiB for `K=256`.
- Exact out-of-place sort: one output and one reusable workspace, `16*N` bytes.
- Bounded-integer fast path may additionally use at most 512 KiB of counts.
- Construction: temporary `8*N` bytes, released before the constructor returns.

Measured results for the current machine are recorded in [RESULTS.md](RESULTS.md).
Post-warm statistics and reference-query results are in
[POST_WARM_RESULTS.md](POST_WARM_RESULTS.md).
