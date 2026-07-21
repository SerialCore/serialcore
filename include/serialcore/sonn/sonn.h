/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_SONN
#define SERIALCORE_SONN_SONN

#include <serialcore/sonn/activaton.h>
#include <serialcore/sonn/nnpool.h>

#define SONN_DEFAULT_MAX_DEGREE 64

typedef struct sonn {
    nnpool_t    *pool;              /* underlying memory pool (capacity and storage only) */

    int         current_neurons;    /* number of neurons currently in use */
    int         max_neurons;
    int         max_degree;
    int         input_dim;
    int         output_dim;

    int         in_start;
    int         in_count;
    int         out_start;
    int         out_count;
} sonn_t;

/* Lifecycle */
sonn_t* sonn_create(int input_dim, int output_dim, int max_neurons, int max_degree, activaton_t type);
void sonn_destroy(sonn_t *s);

/* Neuron operations */
int sonn_add_neuron(sonn_t *s, activaton_t type);
void sonn_remove_neuron(sonn_t *s, int id);

/* Returns the index into the pool's edge arena for `a`'s half (or -1 on failure). */
int sonn_add_edge(sonn_t *s, int a, int b);
void sonn_remove_edge(sonn_t *s, int a, int b);

/* Interior neurons are the ones algorithm layers grow and adapt. */
int sonn_is_interior(sonn_t *s, int id);

/* Queries */
int sonn_get_neighbors(sonn_t *s, int id, int *out, int max_out);

/* Range accessors for the fixed input/output neuron anchors. */
int sonn_get_input_range(sonn_t *s, int *start, int *count);
int sonn_get_output_range(sonn_t *s, int *start, int *count);

/* Read out the current activations of the output neurons into `out`. */
int sonn_get_output(sonn_t *s, float *out);

#endif