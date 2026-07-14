/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NNPOOL
#define SERIALCORE_SONN_NNPOOL

#include <serialcore/sonn/neuron.h>

#include <stddef.h>
#include <stdint.h>

/*
 * SONN Pool — Pure memory pool.
 *
 * Responsibilities (strict):
 *   - At creation time, "fill" the entire memory pool (pre-allocate max_neurons units + edge storage).
 *   - Only maintain capacity information:
 *       - Total capacity (max_neurons)
 *       - Used capacity (used_neurons)
 *       - Available capacity (computable)
 *       - Unit size: one neuron slot occupies input_dim floats (weights)
 *   - Provide direct access to raw storage (neurons, weight_block, edges, adj, degrees).
 *   - Provide basic memory access and query operations.
 *
 * Pool **does not contain** any neuron operations (add/remove neurons).
 * All neural network configuration and operations are placed in the Network layer.
 */

#define SONN_DEGREE_MAX 64

typedef struct nnpool {
    /* Raw storage (fully pre-allocated at creation) */
    neuron_t *neurons;              /* [max_neurons] */
    float    *weights;              /* [max_neurons * input_dim] — weights for all neurons */
    float    *biases;               /* [max_neurons] — scalar bias (offset) for each neuron */
    edge_t   *edges;                /* [max_edges] */
    int      *adj;                  /* [max_neurons][max_degree] — adjacency list: neighbor IDs for each neuron */
    int      *degrees;              /* current "degree" = number of neighbors this neuron currently has (0..max_degree) */

    /* Capacity information */
    int      max_neurons;           /* total capacity */
    int      used_neurons;          /* number of claimed units */
    int      input_dim;             /* size of each unit (number of weights) */
    int      max_degree;            /* maximum neighbors per neuron (fan-out limit) */
    int      max_edges;

    /* Slot reuse support (free list for released slots) */
    int      allocated;     /* high-water mark: next new slot to consider (0..max) */
    int     *free_slots;    /* stack of reusable slot ids */
    int      free_count;
} nnpool_t;

/* Lifecycle */
nnpool_t* nnpool_create(int max_neurons, int input_dim);
void nnpool_destroy(nnpool_t *p);

/* Low-level storage query */
int nnpool_find_edge_slot(nnpool_t *p, int from, int to);

/* Claim a raw slot from available capacity.
 * Returns [0, max_neurons), or -1.
 * This is only memory allocation; the Network layer is responsible for interpreting it as a "neuron".
 */
int nnpool_acquire_slot(nnpool_t *p);

/* Return a raw slot back to the pool. */
void nnpool_release_slot(nnpool_t *p, int id);

/* Basic memory access / query (no neuron semantics) */
static inline neuron_t* nnpool_get_neuron(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return &p->neurons[id];
}

static inline float* nnpool_get_weights(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->weights + (size_t)id * p->input_dim;
}

static inline int* nnpool_adjacency_row(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->adj + (size_t)id * p->max_degree;
}

#endif
