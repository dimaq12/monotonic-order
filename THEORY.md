# IdealOrder: Mathematical Specification

This document defines the exact operator implemented by `idealorder`, separates
its exact and approximate parts, and records its algebraic, numerical and
information-theoretic contracts.

The package contains **two distinct operators**:

1. a compact projection of an empirical distribution onto `K+1` quantile
   knots and a small set of exact statistics;
2. an exact stable order operator on arbitrary new IEEE-754 `float64` arrays.

The fitted distribution is not used to sort new arrays. The two operators are
packaged together because they expose complementary order information, not
because sorting is learned from the training distribution.

## 1. Domain and notation

Let

\[
x=(x_0,\ldots,x_{N-1})\in\mathbb R^N
\]

be a finite training sample. Training currently requires every value to be a
finite `float64`. Write its nondecreasing order statistics as

\[
x_{(0)}\le x_{(1)}\le\cdots\le x_{(N-1)}.
\]

The empirical probability measure is

\[
\mu_N=\frac1N\sum_{i=0}^{N-1}\delta_{x_i}.
\]

For a probability `q in [0,1]`, the implementation uses the linear sample
quantile convention

\[
p=q(N-1),\qquad
\ell=\lfloor p\rfloor,\qquad
h=\min(\ell+1,N-1),
\]

\[
Q_N(q)=x_{(\ell)}+(p-\ell)(x_{(h)}-x_{(\ell)}).
\]

This is NumPy's default linear quantile convention.

## 2. Compact distribution operator

Choose an integer resolution `K >= 2`. Define uniform probability nodes

\[
u_j=\frac jK,\qquad j=0,\ldots,K,
\]

and quantile knots

\[
q_j=Q_N(u_j).
\]

The fitted IdealOrder projection is

\[
\boxed{
\mathcal P_K(\mu_N)
=\left(
N,K,(q_0,\ldots,q_K),Q_N(1/4),Q_N(1/2),Q_N(3/4),\operatorname{MAD}
\right).
}
\]

Here

\[
\operatorname{MAD}
=Q_{|x-Q_N(1/2)|}(1/2).
\]

The original sample and its full sorted copy are destroyed after construction.
Only this projected state remains.

### 2.1 Stored exact functionals

The following values are exact for the supplied finite training sample under
the quantile convention above:

\[
\min x,\quad \max x,\quad Q_1,\quad Q_2,\quad Q_3,\quad
\operatorname{IQR}=Q_3-Q_1,\quad \operatorname{MAD}.
\]

They are exact because they are computed from the temporary fully ordered
sample during fitting and stored explicitly. Reading them later is `O(1)`.

This is precomputation, not a claim that the mathematical statistic itself can
be obtained from an arbitrary new array in constant time.

### 2.2 Approximate quantile operator

For an arbitrary `u in [0,1]`, let

\[
t=Ku,\qquad j=\min(\lfloor t\rfloor,K-1),\qquad \alpha=t-j.
\]

The fitted quantile is the piecewise-linear interpolant

\[
\boxed{
\widehat Q_K(u)=(1-\alpha)q_j+\alpha q_{j+1}.
}
\]

At every stored probability node it is exact:

\[
\widehat Q_K(j/K)=Q_N(j/K).
\]

Between knots it is an approximation to the full empirical quantile curve.

### 2.3 Approximate rank operator

For a query value `y`, locate the largest knot index `j` satisfying

\[
q_j\le y,
\]

clamped to a valid interval. If `q_{j+1}>q_j`, define

\[
\boxed{
\widehat F_K(y)
=\frac jK+rac1K\frac{y-q_j}{q_{j+1}-q_j}.
}
\]

The endpoints are fixed as

\[
\widehat F_K(y)=0\quad(y\le q_0),
\qquad
\widehat F_K(y)=1\quad(y\ge q_K).
\]

If neighboring knots coincide, the implementation returns the lower
probability coordinate for that selected flat interval. Thus `rank()` is a
monotone **distribution coordinate**, not an exact count of training elements.

For strictly increasing knots, both the fitted coordinate and the generalized
inverse of the full quantile curve lie inside the same probability cell, so
the knot discretization contributes at most

\[
\frac1K
\]

of normalized rank uncertainty, up to the chosen finite-sample endpoint/rank
convention. For empirical atoms and repeated values, a point has a rank
interval whose width equals its atom mass; that width can exceed `1/K`.
Consequently no unconditional `1/K` point-rank bound is claimed for arbitrary
tie-heavy distributions.

## 3. Storage and fitting complexity

The persistent C state contains a fixed header and `K+1` doubles:

\[
\boxed{M_{\rm model}=\Theta(K),}
\]

independent of training size `N`. With `K=256` the measured state is 2,128
bytes.

Construction currently uses comparison sorting:

\[
T_{\rm fit}=O(N\log N),\qquad
M_{\rm temporary}=O(N).
\]

The temporary array is released before construction returns. Scalar reads are
`O(1)`, approximate quantiles are `O(1)`, and scalar ranks use binary search
over knots:

\[
T_{\rm rank}=O(\log K).
\]

For `M` rank queries the current implementation costs `O(M log K)`.

## 4. Exact IEEE-754 order operator

Let `b(x)` be the 64-bit payload of a `float64`. For every non-NaN value define
the unsigned order key

\[
\tau(x)=
\begin{cases}
\neg b(x),&\text{if the sign bit is 1},\\
b(x)\oplus 2^{63},&\text{if the sign bit is 0}.
\end{cases}
\]

All NaNs are assigned the maximal key

\[
\tau(\operatorname{NaN})=2^{64}-1.
\]

For ordinary non-NaN values,

\[
\boxed{x<y\iff\tau(x)<\tau(y).}
\]

The transform reverses the negative IEEE bit region and moves the positive
region above it. Its resulting order is

```text
-infinity < negative finite < -0 < +0 < positive finite < +infinity < NaNs.
```

Signed zeros therefore occupy distinct total-order positions. All NaNs share
one ordering key and are placed last.

### 4.1 The sorting operator

For an input sequence `a in F64^N`, define

\[
\boxed{
\mathcal S_N(a)=\tau^{-1}\!\left(
\operatorname{stable\_sort}(\tau(a_0),\ldots,\tau(a_{N-1}))
\right).
}
\]

For NaNs the implementation moves the original floating payloads instead of
inverting the common maximal key. Stability therefore preserves the original
relative order and payloads of NaN values.

Stability means that elements with equal order keys preserve their input
order. `-0` and `+0` have different keys, so their mutual ordering is defined
by the total order rather than by equality under ordinary floating comparison.

### 4.2 Radix factorization

The key is factored into five least-significant radix digits:

\[
64=13+13+13+13+12.
\]

The first four passes use `2^13=8192` buckets and the final pass uses `2^12`
buckets. Every pass is a stable counting scatter. The standard LSD-radix
induction gives:

- after pass one, values are ordered by the lowest digit;
- after pass `r`, values are ordered by the lowest `r` digits;
- stability preserves the already established lower-digit order inside every
  new higher-digit bucket;
- after five passes, values are ordered by the complete 64-bit key.

Therefore the output is exact under the declared total-order policy.

### 4.3 Complexity

Because the number of digits and bucket count are constants for `float64`,

\[
\boxed{T_{\rm sort}=\Theta(N),}
\]

with approximately five histogram passes and five scatter passes after key
conversion. Out-of-place Python sorting allocates:

\[
M_{\rm output+workspace}=16N\ \text{bytes},
\]

excluding the input and small fixed histograms.

The asymptotic statement does not guarantee a speed win for every `N`.
Allocation, memory bandwidth, OpenMP startup and NumPy's optimized kernels
matter. Current measurements show NumPy faster around `N=100,000`, while this
implementation wins on the tested machine for general arrays around one
million elements and above.

## 5. Exact fast paths

Before radix execution, the implementation tests several exact cases.

### 5.1 Already ordered input

If keys are nondecreasing, the output is a direct copy:

\[
T=\Theta(N).
\]

### 5.2 Strict reverse order

If keys are nonincreasing and all keys are distinct, reversing the sequence is
the exact sorted output. The distinctness condition prevents reversal from
breaking stability inside equal-key groups.

### 5.3 Bounded integer-valued floats

If every value is an exactly represented integer, no value is negative zero,
and

\[
\max a-\min a\le65535,
\]

counting sort is used:

\[
T=O(N+R),\qquad M=O(R),\qquad R\le65536.
\]

The full array is validated before this path is accepted, so sampling is only
an early rejection optimization and cannot make the result inexact.

### 5.4 Parallel stable radix

With OpenMP enabled and a sufficiently large non-NaN input, every worker owns
a contiguous source slice and private histogram. Global bucket offsets are
assigned in worker order. Since worker slices follow source order, equal keys
from an earlier slice are written before equal keys from a later slice. This
preserves global stability.

OpenMP changes execution strategy, not the mathematical output.

## 6. Derived exact array operators

All array methods below operate on the supplied array, not on discarded
training data.

### `is_sorted(a)`

Tests

\[
\tau(a_i)\le\tau(a_{i+1})
\]

for every adjacent pair. Cost is `O(N)` and storage is `O(1)`.

### `unique(a)`

Computes `S_N(a)` and keeps the first element of every new key. Equality is
key equality:

- `-0` and `+0` are distinct;
- all NaNs form one unique class;
- ordinary identical payloads form one class.

### `bottom_k(a)` and `top_k(a)`

Both currently perform a full exact sort and slice the result. `top_k`
explicitly excludes NaNs. Their current cost is therefore `Theta(N)`, not the
expected `O(N log k)` heap or selection bound.

### `count_between(a,lo,hi)`

Sorts and then uses two binary searches:

\[
\#\{i:lo\le a_i\le hi\}.
\]

The query after sorting is `O(log N)`; the complete one-shot operation remains
`Theta(N)` because it does not retain the sorted array.

## 7. Algebraic properties

For the declared total order, the exact sorting operator satisfies:

### Idempotence

\[
\boxed{\mathcal S_N(\mathcal S_N(a))=\mathcal S_N(a).}
\]

### Multiset preservation

The output is a permutation of the original 64-bit payloads, including NaN
payloads on the NaN-preserving path.

### Permutation invariance of values

For any permutation `pi`, the ordered value multiset is unchanged:

\[
\mathcal S_N(a_\pi)=\mathcal S_N(a),
\]

apart from the intentionally stable relative ordering of equal-key labelled
records if labels are considered external to the numeric array.

### Positive affine equivariance

In exact real arithmetic, for `c>0`,

\[
\mathcal S_N(ca+d)=c\mathcal S_N(a)+d.
\]

For actual `float64`, this holds when the transformation introduces no
overflow, NaN or rounding collision that changes payload-level distinctions.

The compact quantile model has the corresponding real-arithmetic law

\[
Q_{ca+d}(u)=cQ_a(u)+d,
\qquad
\operatorname{MAD}(ca+d)=c\operatorname{MAD}(a).
\]

## 8. Warmed-object semantics

Fitting `IdealOrder(reference)` produces immutable statistics and quantile
knots for **that reference sample**. After fitting:

- `min`, `max`, quartiles, median, IQR and MAD are immediate stored reads;
- `rank` and `quantile` query the compressed reference model;
- `sort(new_array)` exactly sorts any new `float64` array;
- the reference model is not automatically updated by new arrays;
- the reference distribution does not make radix sorting exact or fast—the
  universal key transform does.

Thus “warming” has two meanings that must not be conflated:

1. statistical warming: reference summaries have already been computed;
2. machine warming: code pages, allocator, caches and OpenMP workers have
   already been initialized before a benchmark sample.

NumPy can also provide `O(1)` reads if an application manually caches every
desired statistic. IdealOrder packages a specific immutable cache together
with a compressed CDF and the exact order kernel.

## 9. Information-theoretic boundary

An `O(K)` state with fixed `K << N` cannot answer every exact rank or membership
query about an arbitrary discarded `N`-element sample.

For distinct values selected from a finite universe of size `U`, merely
identifying the stored set requires at least

\[
\log_2\binom UN
\]

bits in the worst case. Exact ranks contain enough information to distinguish
many such sets. When `U` is large relative to `N`, this lower bound grows on
the order of the sample information itself, not `O(K)`.

IdealOrder chooses the explicit trade-off:

\[
\boxed{
O(K)\ \text{persistent memory}
\quad\Longleftrightarrow\quad
\text{approximate arbitrary reference ranks/quantiles}.
}
\]

Exact stored scalar summaries and exact operations on newly supplied arrays do
not violate this boundary because they answer different, finite questions or
read the new data again.

## 10. Exactness ledger

| Operation | Data source | Contract |
|---|---|---|
| `min/max/q1/median/q3/iqr/mad` | fitted reference | exact stored values |
| `quantile(j/K)` | fitted reference | exact stored knot |
| arbitrary `quantile(u)` | fitted reference | piecewise-linear approximation |
| `rank(y)` / `rank_array` | fitted reference | monotone compressed coordinate |
| `sort` / `sort_reverse` | supplied array | exact declared total order |
| `is_sorted` | supplied array | exact |
| `unique` | supplied array | exact key-equivalence classes |
| `top_k` / `bottom_k` | supplied array | exact after full sort |
| `count_between` | supplied array | exact numeric interval count |
| statistics of a later array | later array | not cached unless a new model is fitted |

## 11. API correspondence

The mathematical objects map to Python as follows:

```python
from ideal_order import IdealOrder, sort

model = IdealOrder(reference, n_bins=K)  # P_K(mu_N)
model.quantile(u)                         # Q_hat_K(u)
model.rank(y)                             # F_hat_K(y)
model.median                              # exact cached Q_N(1/2)
sort(values)                              # S_N(values)
```

The C API in `ideal_order.h` exposes the same core without requiring Python.

## 12. Monotonic-key permutation extension

Version 0.3 exposes the key-level operator underlying ordering. For a payload
sequence `z` and a supported monotonic key sequence `kappa(z)`, define

\[
\Pi_\kappa(z)=\operatorname{stable\_argsort}(\kappa(z)).
\]

The ordered payload is

\[
\boxed{\operatorname{Order}_\kappa(z)=z[\Pi_\kappa(z)].}
\]

The native kernel moves `size_t` indices rather than payload objects. This
separates the universal order computation from representation-specific gather
cost. Current exact codecs are:

- `uint64`: identity key;
- `int64`: sign-bit flip;
- `float64`: the IEEE transform `tau` defined above;
- `datetime64/timedelta64`: signed tick order with separate `NaT` placement;
- UUID: two lexicographic `uint64` words representing the full 128-bit integer.

For fixed-width 64-bit keys, stable radix argsort remains `Theta(N)` and uses
two index arrays, or `16N` bytes on a 64-bit platform, excluding input keys.
Arbitrary Python payloads are supported by materializing one supported key per
object; no Python comparator runs inside the native radix passes.

Multiple fields can be composed by stable multiword radix passes. Enum,
variable-width string and spatial codecs remain specified but unimplemented in
`EPIC_MONOTONIC_KEYS.md`.

## 13. Benchmark interpretation

The current recorded benchmark is evidence for one hardware/software regime,
not an asymptotic proof of universal superiority:

- exact general sort at `N=1,000,000`: approximately `1.7x` faster than the
  tested NumPy build;
- reversed input: approximately `4.6x` through the exact reverse fast path;
- `N=100,000`: NumPy remains faster;
- compact reference state at `K=256`: 2,128 bytes versus an 8 MB exact sorted
  cache for one million values;
- bulk approximate ranks: lower memory and faster in the recorded test, with
  nonzero approximation error;
- stored statistics: large apparent speedups only because computation occurred
  during fitting and NumPy was asked to recompute from the raw array.

The honest computational claim is therefore:

\[
\boxed{
\text{a compact immutable order model plus a competitive exact linear-time
float64 sorting kernel},
}
\]

not “all order queries become exact and constant-time” and not “every array is
faster than NumPy.”
