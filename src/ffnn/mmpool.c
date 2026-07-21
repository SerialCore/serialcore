/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/mmpool.h>
#include <serialcore/math/xoshiross.h>

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/* He-initialize the parameter arena */
static void generate_random_parameter(mmpool_t *p)
{
    for (int l = 0; l < p->n_layers; ++l) {
        int inputs  = p->layer_inputs[l];
        int outputs = p->layer_outputs[l];
        int nw      = outputs * inputs;

        float *weights = p->params + p->offs[l];
        float *biases  = weights + nw;

        float scale = sqrtf(2.0f / (float)inputs);
        for (int i = 0; i < nw; ++i) {
            uint64_t r = next();
            float u = (float)((r >> 11) * (1.0 / 9007199254740992.0));
            weights[i] = scale * (2.0f * u - 1.0f);
        }
        for (int o = 0; o < outputs; ++o) {
            biases[o] = 0.0f;
        }
    }
}

mmpool_t* mmpool_create(const int *inputs_arr, const int *outputs_arr, int n_layers)
{
    if (!inputs_arr || !outputs_arr || n_layers <= 0) return NULL;

    mmpool_t *p = (mmpool_t*)calloc(1, sizeof(mmpool_t));
    if (!p) return NULL;

    p->n_layers = n_layers;

    /* Per-layer geometry + offset table. 
     * Build offsets[l] as the prefix sum of `weight_count + bias_count == outputs*inputs + outputs`. */
    p->offs          = (int*)calloc((size_t)n_layers + 1, sizeof(int));
    p->layer_inputs  = (int*)calloc((size_t)n_layers,     sizeof(int));
    p->layer_outputs = (int*)calloc((size_t)n_layers,     sizeof(int));
    if (!p->offs || !p->layer_inputs || !p->layer_outputs) {
        free(p->offs); free(p->layer_inputs); free(p->layer_outputs); free(p);
        return NULL;
    }

    int total = 0;
    for (int l = 0; l < n_layers; ++l) {
        if (inputs_arr[l] <= 0 || outputs_arr[l] <= 0) {
            free(p->offs); free(p->layer_inputs); free(p->layer_outputs); free(p);
            return NULL;
        }
        p->layer_inputs[l]  = inputs_arr[l];
        p->layer_outputs[l] = outputs_arr[l];
        p->offs[l] = total;
        total += outputs_arr[l] * inputs_arr[l] + outputs_arr[l];
    }
    p->offs[n_layers] = total;
    p->param_count = total;

    /* Fill the entire memory pool at creation time. */
    p->params = (float*)calloc((size_t)total, sizeof(float));
    p->grads  = (float*)calloc((size_t)total, sizeof(float));
    if (!p->params || !p->grads) {
        free(p->params); free(p->grads);
        free(p->offs); free(p->layer_inputs); free(p->layer_outputs); free(p);
        return NULL;
    }

    generate_random_parameter(p);
    return p;
}

void mmpool_destroy(mmpool_t *p)
{
    if (!p) return;
    free(p->params);
    free(p->grads);
    free(p->offs);
    free(p->layer_inputs);
    free(p->layer_outputs);
    free(p);
}