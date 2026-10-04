/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_MMPOOL
#define SERIALCORE_FFNN_MMPOOL

#include <stdlib.h>

typedef struct mmpool {
    float *params;          /* [param_count], NULL when param_count is 0 */
    float *grads;           /* [param_count], same length as params */

    /* offs[l] is the start of layer l. offs[n_layers] == param_count.
     * A layer with no parameters has offs[l] == offs[l + 1]. */
    int   *offs;
    int    n_layers;
    int    param_count;
} mmpool_t;

/* counts[l] may be 0. Storage is zeroed; each layer initializes its own block. */
mmpool_t *mmpool_create(const int *counts, int n_layers);
void mmpool_destroy(mmpool_t *p);

static inline float *mmpool_params_at(mmpool_t *p, int l)
{
    if (!p || !p->params || l < 0 || l >= p->n_layers) return NULL;
    if (p->offs[l] == p->offs[l + 1]) return NULL;
    return p->params + p->offs[l];
}

static inline float *mmpool_grads_at(mmpool_t *p, int l)
{
    if (!p || !p->grads || l < 0 || l >= p->n_layers) return NULL;
    if (p->offs[l] == p->offs[l + 1]) return NULL;
    return p->grads + p->offs[l];
}

#endif
