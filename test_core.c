#include "ideal_order.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    const size_t n = 10000u;
    double *x = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    double *tmp = malloc(n * sizeof(double));
    assert(x && out && tmp);
    for (size_t i = 0; i < n; ++i) x[i] = sin((double)i) * (double)(i % 97u);
    IdealOrder *order = ideal_order_create(x, n, 128u);
    assert(order != NULL);
    assert(ideal_order_storage_bytes(order) < 4096u);
    assert(ideal_order_sort(x, n, out, tmp));
    assert(ideal_order_is_sorted(out, n));
    const unsigned long long keys[] = {3u, 1u, 3u, 2u, 1u};
    size_t permutation[5], index_workspace[5];
    assert(ideal_order_argsort_u64(keys, 5u, permutation, index_workspace));
    const size_t expected[] = {1u, 4u, 3u, 0u, 2u};
    for (size_t i = 0; i < 5u; ++i) assert(permutation[i] == expected[i]);
    ideal_order_destroy(order);
    free(tmp); free(out); free(x);
    return 0;
}
