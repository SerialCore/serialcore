/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_SONN
#define SERIALCORE_SONN_SONN

#include <serialcore/sonn/activaton.h>
#include <serialcore/sonn/nnpool.h>

/*
 * SONN implementation — structural / plumbing layer only.
 *
 * This translation unit contains NO self-organizing algorithm. It deliberately
 * exposes only network construction, teardown, and topology queries. The
 * Growing-Neural-Gas algorithm that used to live here has been moved to
 * <serialcore/sonn/gng.h> (src/sonn/gng.c). Other algorithms (SOM, NG, ...) can
 * build on this same primitive layer without being coupled to GNG.
 *
 * Responsibilities kept here:
 *   - Allocate the network with fixed input/output anchors bracketing a free
 *     pool of interior slots.
 *   - Add / remove neurons (interior only; anchors are permanent).
 *   - Add / remove bidirectional edges (with or without weight writes).
 *   - Query adjacency, ranges, neuron IDs, and output activations.
 *
 * All raw memory allocation and slot management is delegated to the Pool.
 */

/* Maximum degree supported by nnpool layouts that don't specify one. Used
 * only as a default by sonn_create(); algorithm layers may pass an explicit
 * value. */
#define SONN_DEFAULT_MAX_DEGREE 64

typedef struct sonn {
    nnpool_t    *pool;             /* underlying memory pool (capacity and storage only) */

    /* Network configuration and state */
    int         current_neurons;      /* number of neurons currently in use */
    int         max_neurons;
    int         max_degree;
    int         input_dim;
    int         output_dim;

    /* Fixed input/output neuron ranges (slot indices in the pool) */
    int         in_start;
    int         in_count;
    int         out_start;
    int         out_count;
} sonn_t;

/* Lifecycle */
sonn_t* sonn_create(int input_dim, int output_dim, int max_neurons, int max_degree);
void sonn_destroy(sonn_t *s);

/* Neuron operations */
int  sonn_add_neuron(sonn_t *s, activaton_t type);
void sonn_remove_neuron(sonn_t *s, int id);

/* Edge operations */
int sonn_add_edge(sonn_t *s, int from, int to, float weight);

/* Add a topology-only edge (no weight written into neuron's weights[]).
 * Use this for GNG / self-organizing networks where neuron weights are prototypes.
 */
int sonn_add_edge_topology(sonn_t *s, int from, int to);

void sonn_remove_edge(sonn_t *s, int from, int to);

/* A neuron slot is "interior" if it is not one of the fixed input anchors nor
 * one of the fixed output anchors. Interior neurons are the ones algorithm
 * layers (GNG, SOM, ...) grow and adapt. */
int sonn_is_interior(sonn_t *s, int id);

/* Queries */
int sonn_get_neighbors(sonn_t *s, int id, int *out, int max_out);
int sonn_get_neuronid(sonn_t *s, int slot);

/* Range accessors for the fixed input/output neuron anchors. */
int sonn_get_input_range(sonn_t *s, int *start, int *count);
int sonn_get_output_range(sonn_t *s, int *start, int *count);

/* Read out the current activations of the output neurons into `out` (length
 * output_dim). Algorithm layers may update output activations directly when
 * processing a sample; no propagation pass is performed because the SONN is
 * shape-driven, not layer-driven.
 */
int sonn_get_output(sonn_t *s, float *out);

/* Total number of neurons currently in use (input + output + interior). */
static inline int sonn_current_neurons(sonn_t *s) { return s ? s->current_neurons : 0; }

/* Number of interior (grown) neurons — i.e. all neurons outside the fixed
 * input/output anchor ranges. */
static inline int sonn_interior_neurons(sonn_t *s)
{
    if (!s) return 0;
    return s->current_neurons - s->in_count - s->out_count;
}

#endif