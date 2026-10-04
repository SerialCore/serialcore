/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NEURON
#define SERIALCORE_SONN_NEURON

#include <serialcore/sonn/activaton.h>

/* Featured input dimensions. Powers of 2 are preferred for cache alignment. */
#define SONN_INPUTDIM_64    64
#define SONN_INPUTDIM_128   128
#define SONN_INPUTDIM_256   256
#define SONN_INPUTDIM_512   512
#define SONN_INPUTDIM_1024  1024

/*
 * Substrate neuron. Algorithm accumulators (GNG error, edge age, readout)
 * do not live here; each algorithm owns those beside the graph.
 * `type` is unit configuration, not a running statistic.
 * `bias` mirrors pool params[0]. `weights` is the prototype, params[1..input_dim].
 * Prototype models (GNG, SOM, Neural Gas) do not read `bias`. It stays on the
 * neuron so the parameter block remains an affine unit for a later algorithm.
 */
typedef struct neuron {
    int         id;
    int         active;
    activaton_t type;
    float       bias;
    float       *weights;
    int         input_dim;
} neuron_t;

/* Undirected adjacency only. Age and other edge attributes belong to the algorithm. */
typedef struct edge {
    int         from;
    int         to;
    int         active;
} edge_t;

/* Activate / initialize a pre-created neuron slot. */
void neuron_activate(neuron_t *n, activaton_t type, float *params, int input_dim);

/* Mark a neuron as inactive and clear its pointer state. */
void neuron_deactivate(neuron_t *n);

#endif
