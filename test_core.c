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
    ideal_order_destroy(order);
    free(tmp); free(out); free(x);
    return 0;
}
