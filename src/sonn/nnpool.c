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

#include <stdlib.h>

nnpool_t* nnpool_create(int max_neurons, int input_dim)
{
    if (max_neurons <= 0) return NULL;

    nnpool_t *p = (nnpool_t*)calloc(1, sizeof(nnpool_t));
    if (!p) return NULL;

    p->max_neurons = max_neurons;
    p->used_neurons = 0;
    p->input_dim = input_dim;
    p->max_degree = SONN_DEGREE_MAX;
    p->max_edges = max_neurons * SONN_DEGREE_MAX;

    /* Fill the entire memory pool at creation time */
    p->neurons = (neuron_t*)calloc(max_neurons, sizeof(neuron_t));

    /* Pre-assign a unique monotonic ID to every neuron slot at creation time.
     * IDs are generated for all slots regardless of active/deactive state.
     * Network has no permission to generate or assign neuron IDs.
     */
    uint64_t id_counter = 0;
    for (int i = 0; i < max_neurons; i++) {
        p->neurons[i].id = id_counter++;
    }
    /* Unified params block: bias at [0], weights at [1 .. input_dim] for each neuron */
    p->params = (float*)calloc((size_t)max_neurons * (input_dim + 1), sizeof(float));
    p->edges = (edge_t*)calloc(p->max_edges, sizeof(edge_t));
    p->degrees = (int*)calloc(max_neurons, sizeof(int));

    /* Free list for fast slot reuse (LIFO) */
    p->free_list = (int*)malloc((size_t)max_neurons * sizeof(int));
    if (!p->free_list) {
        free(p->degrees);
        free(p->edges);
        free(p->params);
        free(p->neurons);
        free(p);
        return NULL;
    }
    /* Initialize free list in reverse so that acquire returns low ids first (0,1,2,...)
     * (matches previous scan-from-0 behavior for initial contiguous allocation)
     */
    for (int i = 0; i < max_neurons; i++) {
        p->free_list[i] = max_neurons - 1 - i;
    }
    p->free_count = max_neurons;

    /* edges are zeroed by calloc:
     * .active == 0 means empty slot.
     * .to will be set only when .active is set.
     * 0 is a valid neuron id, so always test .active, never rely on .to value alone.
     */

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
