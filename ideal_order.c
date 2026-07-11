#include "ideal_order.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define RADIX_BITS 13u
#define RADIX_SIZE (1u << RADIX_BITS)
#define RADIX_MASK (RADIX_SIZE - 1u)
#define RADIX_PASSES 5u

struct IdealOrder {
    size_t n;
    size_t n_bins;
    double min_value;
    double max_value;
    double q1;
    double median;
    double q3;
    double mad;
    double *knots; /* n_bins + 1 approximate quantile knots */
};

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double exact_quantile(const double *x, size_t n, double q) {
    const double position = q * (double)(n - 1u);
    const size_t lo = (size_t)position;
    const size_t hi = lo + 1u < n ? lo + 1u : lo;
    const double fraction = position - (double)lo;
    return x[lo] + fraction * (x[hi] - x[lo]);
}

IdealOrder *ideal_order_create(const double *data, size_t n, size_t n_bins) {
    if (data == NULL || n == 0u || n_bins < 2u) {
        return NULL;
    }

    double *scratch = (double *)malloc(n * sizeof(double));
    IdealOrder *order = (IdealOrder *)calloc(1u, sizeof(*order));
    if (scratch == NULL || order == NULL) {
        free(scratch);
        free(order);
        return NULL;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(data[i])) {
            free(scratch);
            free(order);
            return NULL;
        }
        scratch[i] = data[i];
    }
    qsort(scratch, n, sizeof(double), compare_double);

    order->knots = (double *)malloc((n_bins + 1u) * sizeof(double));
    if (order->knots == NULL) {
        free(scratch);
        free(order);
        return NULL;
    }
    order->n = n;
    order->n_bins = n_bins;
    order->min_value = scratch[0];
    order->max_value = scratch[n - 1u];
    order->q1 = exact_quantile(scratch, n, 0.25);
    order->median = exact_quantile(scratch, n, 0.50);
    order->q3 = exact_quantile(scratch, n, 0.75);
    for (size_t i = 0; i <= n_bins; ++i) {
        const double q = (double)i / (double)n_bins;
        order->knots[i] = exact_quantile(scratch, n, q);
    }

    for (size_t i = 0; i < n; ++i) {
        scratch[i] = fabs(data[i] - order->median);
    }
    qsort(scratch, n, sizeof(double), compare_double);
    order->mad = exact_quantile(scratch, n, 0.50);
    free(scratch);
    return order;
}

void ideal_order_destroy(IdealOrder *order) {
    if (order != NULL) {
        free(order->knots);
        free(order);
    }
}

size_t ideal_order_size(const IdealOrder *order) { return order ? order->n : 0u; }
size_t ideal_order_bins(const IdealOrder *order) { return order ? order->n_bins : 0u; }
size_t ideal_order_storage_bytes(const IdealOrder *order) {
    return order ? sizeof(*order) + (order->n_bins + 1u) * sizeof(double) : 0u;
}
double ideal_order_min(const IdealOrder *o) { return o ? o->min_value : NAN; }
double ideal_order_max(const IdealOrder *o) { return o ? o->max_value : NAN; }
double ideal_order_q1(const IdealOrder *o) { return o ? o->q1 : NAN; }
double ideal_order_median(const IdealOrder *o) { return o ? o->median : NAN; }
double ideal_order_q3(const IdealOrder *o) { return o ? o->q3 : NAN; }
double ideal_order_mad(const IdealOrder *o) { return o ? o->mad : NAN; }

static size_t knot_interval(const IdealOrder *order, double value) {
    size_t lo = 0u;
    size_t hi = order->n_bins + 1u;
    while (lo < hi) {
        const size_t mid = lo + ((hi - lo) >> 1u);
        if (order->knots[mid] <= value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo == 0u) return 0u;
    if (lo > order->n_bins) return order->n_bins - 1u;
    return lo - 1u;
}

double ideal_order_rank(const IdealOrder *order, double value) {
    if (order == NULL || isnan(value)) return NAN;
    if (value <= order->min_value) return 0.0;
    if (value >= order->max_value) return 1.0;
    const size_t j = knot_interval(order, value);
    const double left = order->knots[j];
    const double right = order->knots[j + 1u];
    const double base = (double)j / (double)order->n_bins;
    if (!(right > left)) return base;
    const double fraction = (value - left) / (right - left);
    return base + fraction / (double)order->n_bins;
}

double ideal_order_quantile(const IdealOrder *order, double q) {
    if (order == NULL || isnan(q)) return NAN;
    if (q <= 0.0) return order->min_value;
    if (q >= 1.0) return order->max_value;
    const double position = q * (double)order->n_bins;
    size_t j = (size_t)position;
    if (j >= order->n_bins) j = order->n_bins - 1u;
    const double fraction = position - (double)j;
    return order->knots[j] + fraction * (order->knots[j + 1u] - order->knots[j]);
}

void ideal_order_rank_array(const IdealOrder *order, const double *values,
                            size_t n, double *out) {
    if (order == NULL || values == NULL || out == NULL) return;
    for (size_t i = 0; i < n; ++i) out[i] = ideal_order_rank(order, values[i]);
}

static inline uint64_t order_key_u64(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint64_t exponent = bits & UINT64_C(0x7ff0000000000000);
    const uint64_t mantissa = bits & UINT64_C(0x000fffffffffffff);
    if (exponent == UINT64_C(0x7ff0000000000000) && mantissa != 0u) {
        return UINT64_MAX; /* all NaNs last; stable radix preserves their order */
    }
    return (bits & UINT64_C(0x8000000000000000)) ? ~bits
                                                  : (bits ^ UINT64_C(0x8000000000000000));
}

static inline uint64_t payload_key(const double *slot) {
    uint64_t key;
    memcpy(&key, slot, sizeof(key));
    return key;
}

static inline void store_payload(double *slot, uint64_t key) {
    memcpy(slot, &key, sizeof(key));
}

static inline double value_from_key(uint64_t key) {
    const uint64_t bits = (key & UINT64_C(0x8000000000000000))
                            ? (key ^ UINT64_C(0x8000000000000000))
                            : ~key;
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Exact fast path for dense integer-valued float64 domains. */
static int try_integer_counting_sort(const double *values, size_t n, double *out) {
    const size_t sample = n < 64u ? n : 64u;
    for (size_t i = 0; i < sample; ++i) {
        const double v = values[i];
        if (!isfinite(v) || trunc(v) != v || fabs(v) > 4503599627370496.0 ||
            (v == 0.0 && signbit(v))) return 0;
    }
    int64_t lo = (int64_t)values[0];
    int64_t hi = lo;
    for (size_t i = 0; i < n; ++i) {
        const double v = values[i];
        if (!isfinite(v) || trunc(v) != v || fabs(v) > 4503599627370496.0 ||
            (v == 0.0 && signbit(v))) return 0;
        const int64_t x = (int64_t)v;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        if ((uint64_t)(hi - lo) > UINT64_C(65535)) return 0;
    }
    const size_t range = (size_t)(hi - lo) + 1u;
    size_t *counts = (size_t *)calloc(range, sizeof(size_t));
    if (counts == NULL) return 0;
    for (size_t i = 0; i < n; ++i) ++counts[(size_t)((int64_t)values[i] - lo)];
    size_t w = 0u;
    for (size_t i = 0; i < range; ++i) {
        size_t count = counts[i];
        while (count-- != 0u) out[w++] = (double)(lo + (int64_t)i);
    }
    free(counts);
    return 1;
}

unsigned long long ideal_order_key(double value) {
    return (unsigned long long)order_key_u64(value);
}

int ideal_order_sort(const double *values, size_t n, double *out,
                     double *workspace) {
    if ((n != 0u && (values == NULL || out == NULL || workspace == NULL)) ||
        values == out || out == workspace || values == workspace) {
        return 0;
    }
    if (n == 0u) return 1;

    int has_nan = 0;
    for (size_t i = 0; i < n; ++i) {
        if (isnan(values[i])) { has_nan = 1; break; }
    }
    int ascending = 1;
    int descending = 1;
    int all_distinct = 1;
    uint64_t previous_key = order_key_u64(values[0]);
    for (size_t i = 1; i < n && (ascending || descending); ++i) {
        const uint64_t key = order_key_u64(values[i]);
        if (key < previous_key) ascending = 0;
        if (key > previous_key) descending = 0;
        if (key == previous_key) all_distinct = 0;
        previous_key = key;
    }
    if (ascending) {
        memcpy(out, values, n * sizeof(double));
        return 1;
    }
    if (descending && all_distinct) {
        for (size_t i = 0; i < n; ++i) out[i] = values[n - 1u - i];
        return 1;
    }
    if (!has_nan && try_integer_counting_sort(values, n, out)) return 1;

#ifdef _OPENMP
    /* Stable parallel radix: every worker owns a contiguous input slice and
       a private histogram. Per-worker offsets preserve original order. */
    if (!has_nan && n >= 262144u) {
        int threads = omp_get_max_threads();
        if (threads > 16) threads = 16;
        const int useful = (int)(n / 131072u);
        if (threads > useful) threads = useful;
        if (threads < 2) threads = 2;
        size_t *hist = (size_t *)calloc((size_t)threads * RADIX_SIZE, sizeof(size_t));
        if (hist != NULL) {
            const double *parallel_src = values;
            double *parallel_dst = workspace;
#pragma omp parallel num_threads(threads) shared(parallel_src, parallel_dst, hist)
            {
                const int tid = omp_get_thread_num();
                const size_t begin = n * (size_t)tid / (size_t)threads;
                const size_t end = n * (size_t)(tid + 1) / (size_t)threads;
                size_t *local = hist + (size_t)tid * RADIX_SIZE;
                for (size_t i = begin; i < end; ++i) {
                    store_payload(workspace + i, order_key_u64(values[i]));
                }
#pragma omp barrier
#pragma omp single
                {
                    parallel_src = workspace;
                    parallel_dst = out;
                }
#pragma omp barrier
                for (unsigned pass = 0u; pass < RADIX_PASSES; ++pass) {
                    const unsigned shift = pass * RADIX_BITS;
                    const size_t buckets = pass == RADIX_PASSES - 1u
                                             ? ((size_t)1u << (64u - shift)) : RADIX_SIZE;
                    const uint64_t mask = (uint64_t)(buckets - 1u);
                    memset(local, 0, buckets * sizeof(size_t));
#pragma omp barrier
                    for (size_t i = begin; i < end; ++i) {
                        const size_t digit = (size_t)((payload_key(parallel_src + i) >> shift) & mask);
                        ++local[digit];
                    }
#pragma omp barrier
#pragma omp single
                    {
                        size_t global = 0u;
                        for (size_t b = 0; b < buckets; ++b) {
                            size_t cursor = global;
                            size_t total = 0u;
                            for (int t = 0; t < threads; ++t) {
                                size_t *cell = hist + (size_t)t * RADIX_SIZE + b;
                                const size_t count = *cell;
                                *cell = cursor;
                                cursor += count;
                                total += count;
                            }
                            global += total;
                        }
                    }
#pragma omp barrier
                    for (size_t i = begin; i < end; ++i) {
                        const uint64_t key = payload_key(parallel_src + i);
                        const size_t digit = (size_t)((key >> shift) & mask);
                        store_payload(parallel_dst + local[digit]++, key);
                    }
#pragma omp barrier
#pragma omp single
                    {
                        parallel_src = parallel_dst;
                        parallel_dst = (parallel_dst == workspace) ? out : workspace;
                    }
#pragma omp barrier
                }
                for (size_t i = begin; i < end; ++i) {
                    out[i] = value_from_key(payload_key(parallel_src + i));
                }
            }
            free(hist);
            return 1;
        }
    }
#endif

    if (!has_nan) {
        for (size_t i = 0; i < n; ++i) {
            store_payload(workspace + i, order_key_u64(values[i]));
        }
        const double *key_src = workspace;
        double *key_dst = out;
        size_t key_counts[RADIX_SIZE];
        for (unsigned pass = 0u; pass < RADIX_PASSES; ++pass) {
            const unsigned shift = pass * RADIX_BITS;
            const size_t buckets = pass == RADIX_PASSES - 1u
                                     ? ((size_t)1u << (64u - shift)) : RADIX_SIZE;
            const uint64_t mask = (uint64_t)(buckets - 1u);
            memset(key_counts, 0, buckets * sizeof(size_t));
            for (size_t i = 0; i < n; ++i)
                ++key_counts[(payload_key(key_src + i) >> shift) & mask];
            size_t offset = 0u;
            for (size_t b = 0; b < buckets; ++b) {
                const size_t count = key_counts[b];
                key_counts[b] = offset;
                offset += count;
            }
            for (size_t i = 0; i < n; ++i) {
                const uint64_t key = payload_key(key_src + i);
                const size_t digit = (size_t)((key >> shift) & mask);
                store_payload(key_dst + key_counts[digit]++, key);
            }
            key_src = key_dst;
            key_dst = (key_dst == out) ? workspace : out;
        }
        for (size_t i = 0; i < n; ++i) out[i] = value_from_key(payload_key(key_src + i));
        return 1;
    }

    const double *src = values;
    double *dst = workspace;
    size_t counts[RADIX_SIZE];

    for (unsigned pass = 0u; pass < RADIX_PASSES; ++pass) {
        const unsigned shift = pass * RADIX_BITS;
        const size_t buckets = pass == RADIX_PASSES - 1u
                                 ? ((size_t)1u << (64u - shift)) : RADIX_SIZE;
        const uint64_t mask = (uint64_t)(buckets - 1u);
        memset(counts, 0, buckets * sizeof(size_t));
        for (size_t i = 0; i < n; ++i) {
            const size_t digit = (size_t)((order_key_u64(src[i]) >> shift) & mask);
            ++counts[digit];
        }
        size_t offset = 0u;
        for (size_t b = 0; b < buckets; ++b) {
            const size_t count = counts[b];
            counts[b] = offset;
            offset += count;
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t digit = (size_t)((order_key_u64(src[i]) >> shift) & mask);
            dst[counts[digit]++] = src[i];
        }
        src = dst;
        dst = (dst == workspace) ? out : workspace;
    }
    if (src != out) memcpy(out, src, n * sizeof(double));
    return 1;
}

int ideal_order_sort_inplace(double *values, size_t n, double *workspace) {
    if (n == 0u) return 1;
    if (values == NULL || workspace == NULL || values == workspace) return 0;
    double *second = (double *)malloc(n * sizeof(double));
    if (second == NULL) return 0;
    const int ok = ideal_order_sort(values, n, second, workspace);
    if (ok) memcpy(values, second, n * sizeof(double));
    free(second);
    return ok;
}

int ideal_order_is_sorted(const double *values, size_t n) {
    if (values == NULL && n != 0u) return 0;
    for (size_t i = 1u; i < n; ++i) {
        if (order_key_u64(values[i]) < order_key_u64(values[i - 1u])) return 0;
    }
    return 1;
}

size_t ideal_order_unique_sorted(const double *sorted, size_t n, double *out) {
    if (n == 0u) return 0u;
    if (sorted == NULL || out == NULL) return 0u;
    size_t written = 1u;
    out[0] = sorted[0];
    uint64_t previous = order_key_u64(sorted[0]);
    for (size_t i = 1u; i < n; ++i) {
        const uint64_t key = order_key_u64(sorted[i]);
        if (key != previous) {
            out[written++] = sorted[i];
            previous = key;
        }
    }
    return written;
}

int ideal_order_argsort_u64(const unsigned long long *raw_keys, size_t n,
                            size_t *indices, size_t *workspace) {
    const uint64_t *keys = (const uint64_t *)raw_keys;
    if ((n != 0u && (keys == NULL || indices == NULL || workspace == NULL)) ||
        indices == workspace) return 0;
    if (n == 0u) return 1;

    int ascending = 1;
    for (size_t i = 0u; i < n; ++i) {
        indices[i] = i;
        if (i != 0u && keys[i] < keys[i - 1u]) ascending = 0;
    }
    if (ascending) return 1;

    size_t *src = indices;
    size_t *dst = workspace;
    size_t counts[RADIX_SIZE];
    for (unsigned pass = 0u; pass < RADIX_PASSES; ++pass) {
        const unsigned shift = pass * RADIX_BITS;
        const size_t buckets = pass == RADIX_PASSES - 1u
                                 ? ((size_t)1u << (64u - shift)) : RADIX_SIZE;
        const uint64_t mask = (uint64_t)(buckets - 1u);
        memset(counts, 0, buckets * sizeof(size_t));
        for (size_t i = 0u; i < n; ++i)
            ++counts[(keys[src[i]] >> shift) & mask];
        size_t offset = 0u;
        for (size_t bucket = 0u; bucket < buckets; ++bucket) {
            const size_t count = counts[bucket];
            counts[bucket] = offset;
            offset += count;
        }
        for (size_t i = 0u; i < n; ++i) {
            const size_t index = src[i];
            const size_t digit = (size_t)((keys[index] >> shift) & mask);
            dst[counts[digit]++] = index;
        }
        size_t *swap = src;
        src = dst;
        dst = swap;
    }
    if (src != indices) memcpy(indices, src, n * sizeof(size_t));
    return 1;
}
