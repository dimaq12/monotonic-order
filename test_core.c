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
    const unsigned long long words[] = {
        1u, 0u, 1u, 0u, 1u,
        2u, 3u, 1u, 3u, 1u
    };
    assert(ideal_order_lexargsort_u64(words, 2u, 5u, permutation, index_workspace));
    const size_t lex_expected[] = {1u, 3u, 2u, 4u, 0u};
    for (size_t i = 0; i < 5u; ++i) assert(permutation[i] == lex_expected[i]);
    const unsigned char text[] = {'b', 'a', 'a', 'a', 'b'};
    const size_t offsets[] = {0u, 1u, 2u, 4u, 5u}; /* b, a, aa, b */
    size_t text_permutation[4], text_workspace[4];
    assert(ideal_order_argsort_bytes(text, sizeof(text), offsets, 4u, 0,
                                     text_permutation, text_workspace));
    const size_t text_expected[] = {1u, 2u, 0u, 3u};
    for (size_t i = 0; i < 4u; ++i) assert(text_permutation[i] == text_expected[i]);
    ideal_order_destroy(order);
    free(tmp); free(out); free(x);
    return 0;
}
