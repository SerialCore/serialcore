/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NNPOOL
#define SERIALCORE_SONN_NNPOOL

#include <serialcore/sonn/neuron.h>

#include <stdlib.h>

/* Default maximum degree */
#define SONN_DEFAULT_MAX_DEGREE 64

typedef struct nnpool {
    neuron_t *neurons;              /* [max_neurons] */
    float    *params;               /* [max_neurons * (input_dim + 1)] — bias at [0], weights [1..] for each neuron */
    edge_t   *edges;                /* [max_neurons * max_degree] — edge attributes + connectivity (use .to when .active) */
    int      *degrees;              /* current "degree" = number of neighbors this neuron currently has (0..max_degree) */

    int      max_neurons;           /* total capacity */
    int      used_neurons;          /* number of claimed units */
    int      max_degree;            /* maximum neighbors per neuron */
    int      max_edges;             /* max_edges = max_neurons * max_degree */
    int      input_dim;             /* size of each neuron's weight vector */

    int     *free_list;
    int      free_count;
} nnpool_t;

/* Lifecycle */
nnpool_t* nnpool_create(int max_neurons, int input_dim, int max_degree);
void nnpool_destroy(nnpool_t *p);

/* Claim a raw slot from free list. */
int nnpool_acquire_slot(nnpool_t *p);

/* Return a raw slot back to the free list. */
void nnpool_release_slot(nnpool_t *p, int id);

/* Find a edge slot between from and to */
int nnpool_find_edge_slot(nnpool_t *p, int from, int to);

/* Returns a pointer to the neuron slot with id */
static inline neuron_t* nnpool_get_neuron(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return &p->neurons[id];
}

/* Returns pointer to the full parameter block for a neuron: params[0] = bias, params[1...] = weights */
static inline float* nnpool_get_params(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->params + id * (p->input_dim + 1);
}

/* Returns a pointer to the row of edge_t slots for a neuron. */
static inline edge_t* nnpool_edge_row(nnpool_t *p, int id)
{
    if (!p || id < 0 || id >= p->max_neurons) return NULL;
    return p->edges + id * p->max_degree;
}

#endif
