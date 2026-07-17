/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * SONN Pool implementation — pure memory management.
 *
 * Strict constraints:
 *   - At creation time, "fill" the entire memory pool (one-time full allocation).
 *   - Use free_list for O(1) slot acquire/release (no scans).
 *   - Maintain capacity (used_neurons, free_count).
 *   - Provide direct access to raw storage and the most basic storage queries.
 *
 * Pool **does not contain**:
 *   - Neuron add/remove operations (those belong to the Network layer with semantics)
 *   - Activation functions, active state, edge rules, etc.
 */

#include <serialcore/sonn/nnpool.h>
#include <serialcore/math/xoshiross.h>

#include <stdlib.h>
#include <stdint.h>

/* Initialize the parameters using the original next() / jump() symbols
 * provided by xoshiross.h (implemented in xoshiro256ss.c).
 *
 * We "seed" by advancing the generator (mixing next() + jump()) a
 * number of steps derived from the pool seed and the neuron id.
 */
static void generate_random_parameter(nnpool_t *p)
{
    int max_neurons = p->max_neurons;
    int input_dim = p->input_dim;
    int seed = p->seed;

    for (int i = 0; i < max_neurons; i++) {
        uint64_t nid = (uint64_t)p->neurons[i].id;

        /* Mix using the original symbols */
        uint32_t mix = (uint32_t)(seed ^ nid) & 0xFFu;
        for (uint32_t k = 0; k < mix; k++) {
            (void)next();
        }
        if (nid & 1) {
            jump();   /* use original jump symbol */
        }

        float *slot = p->params + i * (input_dim + 1);

        /* bias - small */
        uint64_t r = next();
        float f = (r >> 11) * (1.0f / 9007199254740992.0f);
        slot[0] = (f * 2.0f - 1.0f) * 0.01f;

        /* weights */
        for (int w = 0; w < input_dim; w++) {
            r = next();
            f = (r >> 11) * (1.0f / 9007199254740992.0f);
            slot[w + 1] = (f * 2.0f - 1.0f) * 0.1f;
        }
    }
}

nnpool_t* nnpool_create(int max_neurons, int input_dim, int seed)
{
    if (max_neurons <= 0) return NULL;

    nnpool_t *p = (nnpool_t*)calloc(1, sizeof(nnpool_t));
    if (!p) return NULL;

    p->max_neurons = max_neurons;
    p->used_neurons = 0;
    p->max_degree = SONN_DEGREE_MAX;
    p->max_edges = max_neurons * SONN_DEGREE_MAX;
    p->input_dim = input_dim;
    p->seed = seed;

    /* Fill the entire memory pool at creation time */
    p->neurons = (neuron_t*)calloc(max_neurons, sizeof(neuron_t));

    /* Pre-assign a unique monotonic ID to every neuron slot at creation time */
    for (int i = 0; i < max_neurons; i++) {
        p->neurons[i].id = i;
    }

    /* Unified params block: bias at [0], weights at [1 .. input_dim] for each neuron */
    p->params = (float*)calloc(max_neurons * (input_dim + 1), sizeof(float));  
    generate_random_parameter(p);
    
    p->edges = (edge_t*)calloc(p->max_edges, sizeof(edge_t));
    p->degrees = (int*)calloc(max_neurons, sizeof(int));
    p->free_list = (int*)calloc(max_neurons, sizeof(int));

    /* Initialize free list in reverse so that acquire returns low ids first (0,1,2,...) */
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

    /* Pop from free list (LIFO reuse) */
    int slot = p->free_list[--p->free_count];
    p->used_neurons++;
    return slot;
}

void nnpool_release_slot(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return;
    if (p->free_count >= p->max_neurons) return;

    /* Push back to free list for reuse */
    p->free_list[p->free_count++] = id;
    if (p->used_neurons > 0) {
        p->used_neurons--;
    }
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
