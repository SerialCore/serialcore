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
 *   - Only maintain capacity information (total capacity, used capacity).
 *   - Provide direct access to raw storage and the most basic storage queries.
 *
 * Pool **does not contain**:
 *   - Neuron add/remove operations (those belong to the Network layer with semantics)
 *   - Activation functions, active state, edge rules, etc.
 */

#include <serialcore/sonn/nnpool.h>

#include <stdlib.h>
#include <string.h>

nnpool_t* nnpool_create(int max_neurons, int input_dim)
{
    if (max_neurons <= 0) return NULL;

    nnpool_t *p = (nnpool_t*)calloc(1, sizeof(nnpool_t));
    if (!p) return NULL;

    p->max_neurons = max_neurons;
    p->used_neurons = 0;
    p->input_dim = input_dim;
    p->max_degree = SONN_DEGREE_MAX;
    p->max_edges = max_neurons * p->max_degree;

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
    p->weights = (float*)calloc((size_t)max_neurons * input_dim, sizeof(float));
    p->biases = (float*)calloc(max_neurons, sizeof(float));   /* one bias per neuron */
    p->edges = (edge_t*)calloc(p->max_edges, sizeof(edge_t));
    p->adj = (int*)calloc((size_t)max_neurons * p->max_degree, sizeof(int));
    p->degrees = (int*)calloc(max_neurons, sizeof(int));

    /* The adjacency list must be initialized to -1 to mean "no connection".
     * 0 is a valid slot index; without this initialization, get_neighbors would return wrong results.
     */
    for (int i = 0; i < max_neurons * p->max_degree; i++) {
        p->adj[i] = -1;
    }

    return p;
}

void nnpool_destroy(nnpool_t *p)
{
    if (!p) return;
    free(p->neurons);
    free(p->weights);
    free(p->biases);
    free(p->edges);
    free(p->adj);
    free(p->degrees);
    free(p);
}

int nnpool_acquire_slot(nnpool_t *p)
{
    if (!p) return -1;
    if (p->used_neurons >= p->max_neurons) return -1;

    int id = p->used_neurons;
    p->used_neurons++;

    /* Reset bias for the newly claimed slot (weights are handled by caller) */
    if (p->biases) {
        p->biases[id] = 0.0f;
    }

    return id;
}

void nnpool_release_slot(nnpool_t *p, int id)
{
    if (!p) return;
    if (id < 0 || id >= p->max_neurons) return;

    if (p->used_neurons > 0) {
        p->used_neurons--;
    }
}

int nnpool_find_edge_slot(nnpool_t *p, int from, int to)
{
    if (!p || from < 0 || to < 0) return -1;
    if (from >= p->max_neurons || to >= p->max_neurons) return -1;

    int *row = nnpool_adjacency_row(p, from);
    if (!row) return -1;

    for (int s = 0; s < p->max_degree; s++) {
        if (row[s] == to) return s;
    }
    return -1;
}
