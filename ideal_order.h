#ifndef IDEAL_ORDER_H
#define IDEAL_ORDER_H

#include <stddef.h>

#if defined(_WIN32)
#define IDEAL_ORDER_API __declspec(dllexport)
#else
#define IDEAL_ORDER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IdealOrder IdealOrder;

/* The training array is used only during construction and is not retained. */
IDEAL_ORDER_API IdealOrder *ideal_order_create(const double *data, size_t n, size_t n_bins);
IDEAL_ORDER_API void ideal_order_destroy(IdealOrder *order);

IDEAL_ORDER_API size_t ideal_order_size(const IdealOrder *order);
IDEAL_ORDER_API size_t ideal_order_bins(const IdealOrder *order);
IDEAL_ORDER_API size_t ideal_order_storage_bytes(const IdealOrder *order);

IDEAL_ORDER_API double ideal_order_min(const IdealOrder *order);
IDEAL_ORDER_API double ideal_order_max(const IdealOrder *order);
IDEAL_ORDER_API double ideal_order_q1(const IdealOrder *order);
IDEAL_ORDER_API double ideal_order_median(const IdealOrder *order);
IDEAL_ORDER_API double ideal_order_q3(const IdealOrder *order);
IDEAL_ORDER_API double ideal_order_mad(const IdealOrder *order);

/* Compressed-reference queries: approximate by construction. */
IDEAL_ORDER_API double ideal_order_rank(const IdealOrder *order, double value);
IDEAL_ORDER_API double ideal_order_quantile(const IdealOrder *order, double q);
IDEAL_ORDER_API void ideal_order_rank_array(const IdealOrder *order, const double *values,
                                             size_t n, double *out);

/* Exact operations on the supplied array. NaNs are placed last. */
IDEAL_ORDER_API int ideal_order_sort(const double *values, size_t n, double *out,
                                     double *workspace);
IDEAL_ORDER_API int ideal_order_sort_inplace(double *values, size_t n, double *workspace);
IDEAL_ORDER_API int ideal_order_is_sorted(const double *values, size_t n);
IDEAL_ORDER_API size_t ideal_order_unique_sorted(const double *sorted, size_t n, double *out);

/* IEEE-754 total-order key used by the exact radix operator. */
IDEAL_ORDER_API unsigned long long ideal_order_key(double value);

#ifdef __cplusplus
}
#endif
#endif
