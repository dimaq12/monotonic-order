# Epic: Universal Ordering by Monotonic Keys

**Target:** `idealorder` 0.2–0.5

**Status:** numeric Phase 1 shipped in `0.2.0`; multiword/UUID/datetime core
shipped in `0.3.0`; remaining codecs planned

**Core rule:** arbitrary payloads are never compared in the radix kernel; the
caller supplies or derives an order-preserving key, and IdealOrder returns a
stable permutation.

## 1. Objective

Generalize the exact `float64` order kernel to arbitrary data without giving up
its central computational property.

For a payload domain `X` with intended total preorder `preceq_X`, require an
encoder

\[
\kappa:X\rightarrow(\mathbb U_{64})^w
\]

such that

\[
\boxed{
a\prec_X b
\iff
\kappa(a)<_{\mathrm{lex}}\kappa(b).
}
\]

Equal keys define an equivalence class. Stable ordering preserves input order
inside each class.

The universal operator is then

\[
\boxed{
\Pi_\kappa(x)
=\operatorname{stable\_argsort}(\kappa(x_0),\ldots,\kappa(x_{N-1})).
}

Payload ordering is only application of the permutation:

\[
\operatorname{Order}_\kappa(x)=x[\Pi_\kappa(x)].
\]

## 2. Why a key, not a comparator

A generic comparator is sufficient for correctness but forces comparison
sorting and its worst-case lower bound

\[
\Omega(N\log N).
\]

A Python comparator also crosses the Python/C boundary repeatedly. A key
callable is acceptable because it is evaluated exactly once per payload; all
subsequent ordering work happens on contiguous fixed-width key arrays.

The epic therefore has two explicit paths:

- radix path: supported monotonic keys, stable fixed-width ordering;
- fallback path: Python `sorted`, clearly reported as comparison-based.

No arbitrary comparator enters the hot C kernel.

## 3. Public API

### 3.1 Primitive permutation

```python
from ideal_order import radix_argsort

permutation = radix_argsort(keys)
ordered_keys = keys[permutation]
```

Proposed signature:

```python
radix_argsort(
    keys,
    *,
    descending: bool = False,
    nulls: Literal["first", "last", "error"] = "last",
) -> np.ndarray  # dtype=np.uintp
```

Supported first: `uint64`, `int64`, `float64`, and fixed-width multiword keys.

### 3.2 Apply a permutation

```python
ordered = apply_order(payload, permutation, axis=0)
```

Contracts:

- NumPy input returns a NumPy array;
- Python sequence input returns a list;
- permutation is validated once at the public boundary;
- the native kernel never copies arbitrary Python objects.

### 3.3 Order payload by an existing key array

```python
ordered = order_by(payload, keys=timestamps)
```

Equivalent to:

```python
apply_order(payload, radix_argsort(timestamps))
```

### 3.4 Order Python objects by a key callable

```python
ordered = order_by(events, key=lambda event: event.timestamp)
```

The callable is invoked exactly `N` times to materialize a supported primitive
or composite key. It is not invoked during radix passes.

### 3.5 Composite records

```python
permutation = radix_lexargsort(
    year,
    month,
    priority,
    identifier,
    descending=(False, False, True, False),
)
```

Lexicographic fields are processed from the least significant field to the
most significant field using stable radix passes.

### 3.6 Explicit codecs

```python
keys = encode_keys(values, codec="datetime64")
keys = encode_keys(uuids, codec="uuid")
keys = encode_keys(points, codec=Morton2D(bounds=..., bits=32))
```

Every codec documents the order it defines. Spatial curve keys are not called
ordinary numerical monotonic maps.

## 4. Internal representation

Use an internal key batch:

```python
KeyBatch(
    words: tuple[np.ndarray, ...],  # uint64, most-significant word first
    null_mask: np.ndarray | None,
    exact: bool,
    order_name: str,
)
```

Requirements:

- every word is contiguous and length `N`;
- words use host `uint64` after explicit endian conversion;
- nullness is separate from value bits;
- descending order transforms value words but does not accidentally reverse
  the requested null placement;
- an `exact=False` codec must expose why information was lost, for example
  coordinate quantization.

## 5. Native C primitives

### 5.1 Stable `uint64` argsort

```c
int ideal_order_argsort_u64(
    const uint64_t *keys,
    size_t n,
    size_t *indices,
    size_t *workspace
);
```

The kernel initializes `indices[i]=i` and stably scatters indices according to
key digits. Payload values never move during radix passes.

### 5.2 Multiword argsort

```c
int ideal_order_lexargsort_u64(
    const uint64_t *const *words,
    size_t n_words,
    size_t n,
    size_t *indices,
    size_t *workspace
);
```

Process the least significant word first. Inside each 64-bit word, process
13-bit radix digits from least significant to most significant.

For fixed word count `w`:

\[
T=\Theta(wN),\qquad M=2N\,\operatorname{sizeof}(\texttt{size_t}).
\]

### 5.3 Optional payload gather

The C core should not initially support arbitrary record strides or Python
objects. NumPy gathering is already optimized and keeps the native ABI small.
Add a typed contiguous gather only after profiling proves it useful.

## 6. Type roadmap

## 6.1 `uint64`

Encoder:

\[
\kappa(x)=x.
\]

This is the foundational kernel and reference implementation.

Acceptance:

- bit-for-bit agreement with `np.argsort(kind="stable")`;
- stability under duplicates;
- empty, singleton, ascending, descending and all-equal arrays;
- serial and OpenMP permutations identical.

## 6.2 `int64`

For two's-complement payload `b(x)`, encode

\[
\boxed{\kappa(x)=b(x)\oplus2^{63}.}
\]

This moves negative integers below nonnegative integers while preserving order
inside both regions.

Acceptance includes `INT64_MIN`, `-1`, `0`, `INT64_MAX` and overflow-free
conversion tests.

## 6.3 `float64`

Reuse the existing IEEE transform

\[
\kappa(x)=
\begin{cases}
\neg b(x),&\operatorname{signbit}(x)=1,\\
b(x)\oplus2^{63},&\operatorname{signbit}(x)=0.
\end{cases}
\]

NaN placement is represented by the null policy rather than silently mixed
with descending-value inversion.

Compatibility contract:

- existing `sort()` output does not change;
- `radix_argsort(x)` followed by `x[p]` equals `sort(x)`;
- `-0 < +0` under the declared total order;
- NaN payload order is stable.

## 6.4 Date and time

Initial scope: NumPy `datetime64` and `timedelta64` with a single normalized
unit.

Encoder:

1. cast to the requested common unit;
2. reinterpret the tick count as `int64`;
3. apply the signed-integer transform;
4. handle `NaT` through the null mask.

Risks:

- unit conversion overflow;
- timezone-aware Python `datetime` versus naive `datetime`;
- ambiguous daylight-saving local times.

Policy:

- Python datetimes must be normalized to UTC ticks before native ordering;
- mixing aware and naive datetimes raises;
- no implicit local timezone is read from the machine.

## 6.5 UUID

A UUID is an unsigned 128-bit value

\[
u=(u_{\rm high},u_{\rm low}).
\]

Encode as two big-endian-order `uint64` words and lexicographically sort high
then low. The radix execution order is low word followed by high word.

Default order matches `UUID.int`, not textual locale ordering.

## 6.6 Enum

Default encoder uses the explicit integer `.value` only when every member has
an integer value and the user confirms that value order is semantic order.

For string-valued or unordered enums, require an explicit rank mapping:

```python
EnumCodec({Priority.LOW: 0, Priority.NORMAL: 1, Priority.HIGH: 2})
```

Declaration order must not be silently treated as semantic order.

## 6.7 Coordinates: Morton and Hilbert

Coordinates usually carry a partial product order or a metric, not one
canonical total order. Morton/Hilbert encoders therefore define a **spatial
curve order**, not a theorem that geometric proximity becomes scalar order.

Required codec parameters:

```python
Morton2D(bounds=((xmin, xmax), (ymin, ymax)), bits=32, clip=False)
Hilbert2D(bounds=..., bits=32, clip=False)
```

Pipeline:

1. validate finite coordinates;
2. normalize using explicit bounds;
3. quantize each axis to `bits`;
4. interleave bits for Morton or apply Hilbert state transitions;
5. emit one or more `uint64` words.

Exactness statement:

- ordering of the quantized curve key is exact;
- mapping continuous coordinates to finite cells is lossy;
- equal quantized cells remain stable;
- clipping is opt-in and reported by the codec.

Morton ships before Hilbert because it is simpler to verify and vectorize.

## 6.8 Byte strings and Unicode strings

Fixed-width bytes can use stable bytewise radix directly. Variable-length
lexicographic order needs an end-of-string sentinel smaller than every byte and
either:

- MSD radix with recursive/iterative bucket refinement; or
- stable LSD passes after length-aware padding with a distinct sentinel.

Public policies:

```python
StringCodec(
    encoding="utf-8",
    normalization=None,  # or NFC/NFKC chosen explicitly
    casefold=False,
    nulls="last",
)
```

The default is deterministic code-point/encoded-byte ordering, not
locale-aware collation. Locale collation requires an external collation-key
generator; IdealOrder can radix-sort those generated bytes but must not invent
linguistic rules.

Milestone order:

1. fixed-width NumPy `S` dtype;
2. variable-length `bytes`;
3. normalized Unicode;
4. external collation keys.

## 6.9 Structured records

Records are ordered by a sequence of field codecs:

```python
spec = RecordOrder(
    field("timestamp", DateTimeCodec(), ascending=True),
    field("priority", Int64Codec(), ascending=False),
    field("id", UUIDCodec(), ascending=True),
)
```

Each field may choose direction and null policy. The combined permutation is a
stable lexicographic composition.

Initial inputs:

- NumPy structured arrays;
- dict of equal-length columns;
- dataclass/object sequences through one-time field extraction.

## 6.10 Arbitrary Python objects

Python objects are supported only through a key extractor whose outputs are
covered by a registered codec:

```python
order_by(objects, key=lambda obj: (obj.timestamp, obj.priority, obj.id))
```

Contracts:

- extractor called exactly once per object;
- exceptions include the failing object index;
- mixed incompatible key types raise;
- keys are materialized before native sorting begins;
- extraction time and radix time are separately measurable.

## 7. Null and missing-value semantics

Every public ordering method accepts

```python
nulls="first" | "last" | "error"
```

Missing values include, by codec:

- floating NaN;
- datetime `NaT`;
- Python `None`;
- optional masked-array entries.

Null placement is independent of ascending/descending value direction. For
example, `descending=True, nulls="last"` must keep missing values last.

All missing values in one field are equal keys for ordering purposes and keep
their original relative order.

## 8. Stability and permutation laws

For every supported codec, test the following laws.

### Valid permutation

\[
\operatorname{sort}(\Pi)=0,1,\ldots,N-1.
\]

### Monotonic output

\[
\kappa(x_{\Pi_i})\le_{\rm lex}\kappa(x_{\Pi_{i+1}}).
\]

### Stability

If `kappa(x_i)=kappa(x_j)` and `i<j`, then `i` appears before `j` in the
permutation.

### Idempotent application

Ordering an already ordered key/payload pair produces the identity
permutation relative to that ordered sequence.

### Composition for records

Multi-field radix output equals stable sorting from the least significant
field through the most significant field.

## 9. Version and delivery plan

## Phase 0 — shared permutation foundation (`0.2.0-alpha`)

- add `ideal_order_argsort_u64` C API;
- expose `radix_argsort(uint64)`;
- expose `apply_order`;
- keep existing float sort unchanged;
- create permutation property tests;
- benchmark indices rather than payload copies.

**Exit gate:** exact stable agreement on `uint64`; no regression greater than
5% in existing `float64 sort` benchmarks.

## Phase 1 — primitive numeric keys (`0.2.0`)

- `uint64`, `int64`, `float64`;
- ascending/descending;
- explicit null policy;
- `order_by(payload, keys=...)`;
- public C API and Python typing.

**Exit gate:** all IEEE/integer edge cases, serial/OpenMP equality, clean
ASAN/UBSAN and wheel tests on Linux/macOS/Windows.

## Phase 2 — multiword and records (`0.3.0`)

- `radix_lexargsort`;
- per-field direction/null rules;
- UUID;
- datetime/timedelta;
- enums;
- NumPy structured records.

**Exit gate:** equality with trusted stable lexicographic references for
random and adversarial record tables.

## Phase 3 — strings (`0.4.0`)

- fixed-width bytes;
- variable-length bytes;
- explicit Unicode normalization/case policy;
- external collation-key adapter.

**Exit gate:** prefix, empty-string, embedded-zero, long-common-prefix,
non-ASCII and normalization controls.

## Phase 4 — spatial keys (`0.5.0`)

- Morton 2D/3D;
- spatial quantization report;
- Hilbert 2D after independent reference verification;
- multiword output when dimension × bits exceeds 64.

**Exit gate:** exhaustive small-grid curve order and round-trip cell tests;
clear separation between exact curve ordering and lossy coordinate
quantization.

## 10. Test matrix

### Deterministic edge tests

- sizes `0,1,2`;
- all equal;
- ascending and descending;
- alternating extrema;
- high duplicate rate;
- every primitive minimum/maximum;
- NaN payloads, signed zero, infinities;
- `NaT`, `None`, empty bytes, prefix strings;
- UUID high/low carry boundaries;
- mixed record directions and null policies.

### Property tests

For seeded randomized arrays:

- permutation validity;
- monotonic encoded keys;
- stable equal-key indices;
- agreement with NumPy/Python reference;
- serial versus OpenMP identity;
- argsort/gather equals direct exact sort for `float64`.

### Cross-platform tests

- CPython 3.9 and current stable;
- Linux GCC and Clang;
- macOS Clang without mandatory OpenMP;
- Windows MSVC;
- little-endian primary support;
- explicit rejection or conversion tests for non-native endian arrays.

### Sanitizers and fuzzing

- ASAN/UBSAN C harness;
- random sizes around radix thresholds;
- malformed strides and noncontiguous Python arrays;
- multiplication/size overflow checks;
- allocation-failure paths where feasible.

## 11. Benchmark contract

Measure key extraction and ordering separately.

```text
total time = encode time + argsort time + payload gather time
```

Datasets:

- primitive keys at `1e5`, `1e6`, `5e6`;
- duplicate ratios `0%, 50%, 99%`;
- two-, four- and eight-field records;
- UUID;
- datetime;
- short and long strings;
- Python objects with cheap and expensive key functions;
- Morton/Hilbert point clouds.

Competitors:

- `np.argsort(kind="stable")`;
- `np.lexsort`;
- Python `sorted(key=...)`;
- specialized string sorting where available.

Report:

- median and dispersion;
- peak memory;
- key materialization memory;
- permutation-only versus gathered payload;
- serial and OpenMP;
- break-even `N`;
- exactness/quantization status.

No release claim may compare radix permutation alone against a competitor that
also performs expensive Python key extraction unless the extraction cost is
reported separately.

## 12. Performance budgets

Initial gates on the reference Linux machine:

- `uint64 radix_argsort(1M)` no slower than `1.15x` the time of current
  `float64` radix ordering after accounting for index movement;
- existing `float64 sort(1M)` regression below 5%;
- peak argsort workspace at most `16N + O(radix buckets)` bytes on 64-bit
  platforms, excluding user keys and returned permutation;
- Python key callable invoked exactly `N` times;
- no hidden object comparisons after key materialization.

These are engineering gates, not universal performance claims. They may be
recalibrated with recorded hardware evidence.

## 13. Risks and decisions

| Risk | Decision |
|---|---|
| Comparator demand | provide documented Python fallback, keep it outside C radix |
| Payload copying dominates | make permutation the primary product |
| Descending breaks null policy | encode nullness as a separate primary field |
| UUID/string endian bugs | canonicalize words/bytes before native call |
| Spatial key called “natural order” | explicitly name it curve order |
| Coordinate quantization hidden | codec returns metadata and `exact=False` |
| Unicode expectations vary | normalization/case/locale must be explicit |
| Object key extraction dominates | benchmark extraction separately |
| Multiword memory grows | stream fields where profiling justifies it later |
| API becomes too broad | codecs remain modular; core only knows uint64 words |

## 14. Non-goals

- no universal acceleration for an arbitrary comparator;
- no claim that hashes preserve order;
- no implicit locale-aware string collation;
- no claim that Morton/Hilbert keys preserve all geometric neighborhoods;
- no automatic semantic ordering of enums;
- no serialization format in this epic;
- no distributed/external-memory sort in versions 0.2–0.5;
- no GPU backend until the CPU permutation contract is frozen.

## 15. Issue breakdown

### Core

- [x] C `argsort_u64` serial kernel
- [ ] C stable OpenMP index scatter
- [x] C multiword lexargsort
- [ ] overflow/allocation guards
- [ ] C API tests and sanitizers

### Numeric Python API

- [x] `radix_argsort`
- [x] `apply_order`
- [x] `order_by(keys=...)`
- [x] signed integer codec
- [ ] refactor float codec around shared permutation laws
- [x] direction and null policies

### Composite codecs

- [x] `radix_lexargsort`
- [x] UUID codec
- [x] datetime/timedelta codec
- [ ] enum rank codec
- [ ] structured-record adapter
- [x] Python object one-shot key extraction

### Variable-width and spatial codecs

- [ ] fixed-width bytes
- [ ] variable bytes
- [ ] Unicode policy layer
- [ ] Morton 2D/3D
- [ ] Hilbert 2D independent verification

### Delivery

- [ ] type annotations and API reference
- [ ] cross-platform CI expansion to Windows
- [ ] benchmark report per phase
- [ ] migration/release notes
- [ ] wheels for supported platforms

## 16. Definition of epic done

The epic is complete when:

1. arbitrary payloads can be stably ordered by supported monotonic key arrays;
2. primitive, multiword, record, string and spatial codecs have explicit order
   contracts;
3. every exact codec agrees with an independent stable reference;
4. every lossy codec reports its quantization boundary;
5. no Python comparator executes inside the radix kernel;
6. existing `IdealOrder` model and `float64 sort` remain backward compatible;
7. Linux, macOS and Windows wheels pass the same contract tests;
8. benchmark reports include encoding, permutation and gather costs separately;
9. the public documentation never conflates semantic order, lexicographic
   order and spatial curve order.

The architectural invariant is simple:

\[
\boxed{
\text{arbitrary payload}
\xrightarrow{\text{monotonic codec}}
\text{fixed-width key words}
\xrightarrow{\text{stable radix}}
\text{permutation}.
}
\]
