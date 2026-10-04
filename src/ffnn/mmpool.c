/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/mmpool.h>

#include <limits.h>
#include <stdlib.h>

mmpool_t *mmpool_create(const int *counts, int n_layers)
{
    mmpool_t *p;
    long long total = 0;

    if (!counts || n_layers <= 0) return NULL;

    p = (mmpool_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_layers = n_layers;
    p->offs = (int *)calloc((size_t)n_layers + 1, sizeof(int));
    if (!p->offs) {
        free(p);
        return NULL;
    }

    for (int l = 0; l < n_layers; ++l) {
        if (counts[l] < 0 || total > INT_MAX - counts[l]) {
            mmpool_destroy(p);
            return NULL;
        }
        p->offs[l] = (int)total;
        total += counts[l];
    }
    p->offs[n_layers] = (int)total;
    p->param_count = (int)total;

    if (total > 0) {
        p->params = (float *)calloc((size_t)total, sizeof(float));
        p->grads = (float *)calloc((size_t)total, sizeof(float));
        if (!p->params || !p->grads) {
            mmpool_destroy(p);
            return NULL;
        }
    }
    return p;
}

void mmpool_destroy(mmpool_t *p)
{
    if (!p) return;
    free(p->params);
    free(p->grads);
    free(p->offs);
    free(p);
}
