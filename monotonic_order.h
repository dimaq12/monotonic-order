#ifndef MONOTONIC_ORDER_H
#define MONOTONIC_ORDER_H

#include <stddef.h>

#if defined(_WIN32)
#define MONOTONIC_ORDER_API __declspec(dllexport)
#else
#define MONOTONIC_ORDER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MonotonicOrder MonotonicOrder;

/* The training array is used only during construction and is not retained. */
MONOTONIC_ORDER_API MonotonicOrder *monotonic_order_create(const double *data, size_t n, size_t n_bins);
MONOTONIC_ORDER_API void monotonic_order_destroy(MonotonicOrder *order);

MONOTONIC_ORDER_API size_t monotonic_order_size(const MonotonicOrder *order);
MONOTONIC_ORDER_API size_t monotonic_order_bins(const MonotonicOrder *order);
MONOTONIC_ORDER_API size_t monotonic_order_storage_bytes(const MonotonicOrder *order);

MONOTONIC_ORDER_API double monotonic_order_min(const MonotonicOrder *order);
MONOTONIC_ORDER_API double monotonic_order_max(const MonotonicOrder *order);
MONOTONIC_ORDER_API double monotonic_order_q1(const MonotonicOrder *order);
MONOTONIC_ORDER_API double monotonic_order_median(const MonotonicOrder *order);
MONOTONIC_ORDER_API double monotonic_order_q3(const MonotonicOrder *order);
MONOTONIC_ORDER_API double monotonic_order_mad(const MonotonicOrder *order);

/* Compressed-reference queries: approximate by construction. */
MONOTONIC_ORDER_API double monotonic_order_rank(const MonotonicOrder *order, double value);
MONOTONIC_ORDER_API double monotonic_order_quantile(const MonotonicOrder *order, double q);
MONOTONIC_ORDER_API void monotonic_order_rank_array(const MonotonicOrder *order, const double *values,
                                             size_t n, double *out);

/* Exact operations on the supplied array. NaNs are placed last. */
MONOTONIC_ORDER_API int monotonic_order_sort(const double *values, size_t n, double *out,
                                     double *workspace);
MONOTONIC_ORDER_API int monotonic_order_sort_inplace(double *values, size_t n, double *workspace);
MONOTONIC_ORDER_API int monotonic_order_is_sorted(const double *values, size_t n);
MONOTONIC_ORDER_API size_t monotonic_order_unique_sorted(const double *sorted, size_t n, double *out);

/* Stable permutation that orders arbitrary unsigned 64-bit monotonic keys. */
MONOTONIC_ORDER_API int monotonic_order_argsort_u64(const unsigned long long *keys, size_t n,
                                             size_t *indices, size_t *workspace);

/* Stable lexicographic permutation. Words are [n_words][n], MS word first. */
MONOTONIC_ORDER_API int monotonic_order_lexargsort_u64(const unsigned long long *words,
                                                size_t n_words, size_t n,
                                                size_t *indices, size_t *workspace);

/* Stable lexicographic order for concatenated variable-length byte strings. */
MONOTONIC_ORDER_API int monotonic_order_argsort_bytes(const unsigned char *data, size_t data_size,
                                               const size_t *offsets, size_t n,
                                               int descending,
                                               size_t *indices, size_t *workspace);

/* Exact Hilbert distance for quantized 2D coordinates, up to 32 bits/axis. */
MONOTONIC_ORDER_API int monotonic_order_hilbert2d_u64(const unsigned long long *x,
                                               const unsigned long long *y,
                                               size_t n, unsigned bits,
                                               unsigned long long *out);

/* IEEE-754 total-order key used by the exact radix operator. */
MONOTONIC_ORDER_API unsigned long long monotonic_order_key(double value);

#ifdef __cplusplus
}
#endif
#endif
