/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NNPOOL
#define SERIALCORE_SONN_NNPOOL

#include <serialcore/sonn/neuron.h>

#include <stdlib.h>

/*
 * SONN Pool — Pure memory pool.
 *
 * Responsibilities (strict):
 *   - At creation time, "fill" the entire memory pool (pre-allocate max_neurons units + edge storage).
 *   - Maintain free_list + free_count for O(1) slot reuse (as recommended in IMPLEMENTATION_GUIDE).
 *   - Only maintain capacity information:
 *       - Total capacity (max_neurons)
 *       - Used capacity (used_neurons)
 *       - Available capacity via free_list
 *       - Unit size: one neuron slot occupies (input_dim + 1) floats
 *   - Provide direct access to raw storage (neurons, params, edges, degrees).
 *   - Provide basic memory access and query operations.
 *
 * Pool **does not contain** any neuron operations (add/remove neurons).
 * All neural network configuration and operations are placed in the Network layer.
 */

#define SONN_DEGREE_MAX 64

typedef struct nnpool {
    /* Raw storage (fully pre-allocated at creation) */
    neuron_t *neurons;              /* [max_neurons] */
    float    *params;               /* [max_neurons * (input_dim + 1)] — bias at [0], weights [1..] for each neuron */
    edge_t   *edges;                /* [max_neurons * max_degree] — edge attributes + connectivity (use .to when .active) */
    int      *degrees;              /* current "degree" = number of neighbors this neuron currently has (0..max_degree) */

    /* Capacity information */
    int      max_neurons;           /* total capacity */
    int      used_neurons;          /* number of claimed units */
    int      input_dim;             /* number of weights per neuron (total params per neuron = input_dim + 1, bias at 0) */
    int      max_degree;            /* maximum neighbors per neuron (fan-out limit) */
    int      max_edges;             /* max_edges = max_neurons * max_degree */

    /* Free list for O(1) slot reuse (stack of available indices) */
    int     *free_list;
    int      free_count;
} nnpool_t;

/* Lifecycle */
nnpool_t* nnpool_create(int max_neurons, int input_dim);
void nnpool_destroy(nnpool_t *p);

/* Claim a raw slot from available capacity using free list.
 * Returns [0, max_neurons), or -1.
 * This is only memory allocation; the Network layer is responsible for interpreting it as a "neuron".
 */
int nnpool_acquire_slot(nnpool_t *p);

/* Return a raw slot back to the pool (push onto free list). */
void nnpool_release_slot(nnpool_t *p, int id);

/* Find a edge slot between from and to */
int nnpool_find_edge_slot(nnpool_t *p, int from, int to);

/* Basic memory access / query (no neuron semantics) */
static inline neuron_t* nnpool_get_neuron(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return &p->neurons[id];
}

/* Returns pointer to the full parameter block for a neuron:
 *   params[0]           = bias
 *   params[1 .. input_dim] = weights
 */
static inline float* nnpool_get_params(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->params + id * (p->input_dim + 1);
}

/* Returns a pointer to the row of edge_t slots for a neuron.
 * Connectivity is stored in e->to when e->active != 0.
 * Use this instead of a separate adjacency list.
 */
static inline edge_t* nnpool_edge_row(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->edges + id * p->max_degree;
}

#endif
