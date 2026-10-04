/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/nnpool.h>
#include <serialcore/math/xoshiross.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/* He-style uniform draw for one neuron's parameter slot (bias + prototype). */
static void fill_random_slot(nnpool_t *p, int id)
{
    int input_dim = p->input_dim;
    float *slot = p->params + (size_t)id * (size_t)(input_dim + 1);
    float scale = sqrtf(2.0f / (float)input_dim);

    slot[0] = 0.0f;
    for (int w = 0; w < input_dim; w++) {
        uint64_t r = next();
        float f = (r >> 11) * (1.0f / 9007199254740992.0f);
        slot[w + 1] = (f * 2.0f - 1.0f) * scale;
    }
}

nnpool_t* nnpool_create(int max_neurons, int input_dim, int max_degree)
{
    if (max_neurons <= 0 || input_dim <= 0 || max_degree <= 0) return NULL;
    if (max_neurons > INT_MAX / max_degree) return NULL;
    if (max_neurons > INT_MAX / (input_dim + 1)) return NULL;

    nnpool_t *p = (nnpool_t*)calloc(1, sizeof(nnpool_t));
    if (!p) return NULL;

    p->max_neurons = max_neurons;
    p->used_neurons = 0;
    p->max_degree = max_degree;
    p->max_edges = max_neurons * max_degree;
    p->input_dim = input_dim;

    p->neurons = (neuron_t*)calloc((size_t)max_neurons, sizeof(neuron_t));
    p->params = (float*)calloc((size_t)max_neurons * (size_t)(input_dim + 1), sizeof(float));
    p->edges = (edge_t*)calloc((size_t)p->max_edges, sizeof(edge_t));
    p->degrees = (int*)calloc((size_t)max_neurons, sizeof(int));
    p->free_list = (int*)calloc((size_t)max_neurons, sizeof(int));
    if (!p->neurons || !p->params || !p->edges || !p->degrees || !p->free_list) {
        nnpool_destroy(p);
        return NULL;
    }

    for (int i = 0; i < max_neurons; i++) {
        p->neurons[i].id = i;
        fill_random_slot(p, i);
    }

    /* Low ids come out first (0, 1, 2, ...). */
    for (int i = 0; i < max_neurons; i++) {
        p->free_list[i] = max_neurons - 1 - i;
    }
    p->free_count = max_neurons;

    return p;
}

void nnpool_destroy(nnpool_t *p)
{
    if (!p) return;
    free(p->free_list);
    free(p->neurons);
    free(p->params);
    free(p->edges);
    free(p->degrees);
    free(p);
}

int nnpool_acquire_slot(nnpool_t *p)
{
    if (!p || p->free_count <= 0) return -1;

    int slot = p->free_list[--p->free_count];
    p->used_neurons++;
    return slot;
}

int nnpool_claim_slot(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return -1;

    for (int i = 0; i < p->free_count; i++) {
        if (p->free_list[i] == id) {
            p->free_list[i] = p->free_list[--p->free_count];
            p->used_neurons++;
            return id;
        }
    }
    return -1;
}

void nnpool_release_slot(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return;

    for (int i = 0; i < p->free_count; i++) {
        if (p->free_list[i] == id) return;
    }
    if (p->free_count >= p->max_neurons) return;

    p->free_list[p->free_count++] = id;
    if (p->used_neurons > 0) {
        p->used_neurons--;
    }

    /* Free slots must not keep the dead unit's prototype or look active. */
    if (p->neurons) {
        p->neurons[id].active = 0;
        p->neurons[id].weights = NULL;
    }
    if (p->params) fill_random_slot(p, id);
}

int nnpool_find_edge_slot(nnpool_t *p, int from, int to)
{
    if (!p || from < 0 || to < 0) return -1;
    if (from >= p->max_neurons || to >= p->max_neurons) return -1;

    edge_t *row = nnpool_edge_row(p, from);
    if (!row) return -1;

    for (int s = 0; s < p->max_degree; s++) {
        if (row[s].active && row[s].to == to) return s;
    }
    return -1;
}
