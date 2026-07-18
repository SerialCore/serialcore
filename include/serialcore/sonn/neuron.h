/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NEURON
#define SERIALCORE_SONN_NEURON

#include <serialcore/sonn/activaton.h>

#include <stdint.h>

/* Featured input dimensions.
 *
 * The caller decides which value to use for each Network at creation time.
 * A single binary can create and operate multiple Networks
 * with different (allowed) input dimensions.
 *
 * Powers of 2 are preferred for cache alignment.
 */
#define SONN_INPUTDIM_64    64
#define SONN_INPUTDIM_128   128
#define SONN_INPUTDIM_256   256
#define SONN_INPUTDIM_512   512
#define SONN_INPUTDIM_1024  1024

typedef struct neuron {
    int         id;
    int         active;
    activaton_t type;
    float       error;          /* for GNG-style error accumulation (keep free for self-organizing use) */
    float       delta;          /* error term for backpropagation */
    float       activation;     /* current activation value after applying activation function */
    float       bias;
    float       *weights;       /* points into pool->params[id*(input_dim+1) + 1] */
    int         input_dim;      /* size of the weights vector (kept as-is per request) */
} neuron_t;

typedef struct edge {
    int         from;
    int         to;
    int         active;
    float       weight;
    float       age;
} edge_t;

/* Activate / initialize a pre-created neuron slot.
 * The ID has already been assigned by the Pool at creation time.
 * Network must not generate or assign neuron IDs.
 *
 * 'params' points to the neuron's unified parameter block in the pool:
 *   params[0] = bias
 *   params[1 .. input_dim] = weights
 *
 * This function initializes both the memory block and the neuron's fields.
 */
void neuron_activate(neuron_t *n, activaton_t type, float *params, int input_dim);

/* Mark a neuron as inactive and clear its pointer state.
 * Does not touch the weight memory itself.
 */
void neuron_deactivate(neuron_t *n);

#endif
