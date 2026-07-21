/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_MMPOOL
#define SERIALCORE_FFNN_MMPOOL

#include <stdlib.h>

typedef struct mmpool {
    float *params;          /* [param_count] — all weights + biases  */
    float *grads;           /* [param_count] — all weight_updates + bias_updates */

    /* Per-layer offset table, length n_layers + 1.
     * offs[l]     = start of layer l's block (weights first, then biases)
     * offs[n_layers] = param_count (sentinel, useful for range iteration)
     * Inside a layer block:
     *   weights      = params + offs[l]
     *   biases       = params + offs[l] + outputs*inputs
     */
    int   *offs;            /* Per-layer offset table, length n_layers + 1 */

    int   *layer_inputs;    /* [n_layers] */
    int   *layer_outputs;   /* [n_layers] */

    int    n_layers;        /* number of layers the pool was built for */
    int    param_count;     /* total floats in `params` (== total in `grads`) */
} mmpool_t;

/* Lifecycle */
mmpool_t* mmpool_create(const int *inputs_arr, const int *outputs_arr, int n_layers);
void mmpool_destroy(mmpool_t *p);

/* Direct accessors. All return NULL on bad index, like nnpool_get_neuron. */
static inline float* mmpool_get_weights(mmpool_t *p, int l)
{
    if (!p || l < 0 || l >= p->n_layers) return NULL;
    return p->params + p->offs[l];
}

static inline float* mmpool_get_biases(mmpool_t *p, int l)
{
    if (!p || l < 0 || l >= p->n_layers) return NULL;
    return p->params + p->offs[l] + p->layer_outputs[l] * p->layer_inputs[l];
}

static inline float* mmpool_get_weight_updates(mmpool_t *p, int l)
{
    if (!p || l < 0 || l >= p->n_layers) return NULL;
    return p->grads + p->offs[l];
}

static inline float* mmpool_get_bias_updates(mmpool_t *p, int l)
{
    if (!p || l < 0 || l >= p->n_layers) return NULL;
    return p->grads + p->offs[l] + p->layer_outputs[l] * p->layer_inputs[l];
}

#endif