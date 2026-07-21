/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NEURON
#define SERIALCORE_SONN_NEURON

#include <serialcore/sonn/activaton.h>

#include <stdint.h>

/* Featured input dimensions. Powers of 2 are preferred for cache alignment. */
#define SONN_INPUTDIM_64    64
#define SONN_INPUTDIM_128   128
#define SONN_INPUTDIM_256   256
#define SONN_INPUTDIM_512   512
#define SONN_INPUTDIM_1024  1024

typedef struct neuron {
    int         id;
    int         active;
    activaton_t type;
    float       bias;
    float       *weights;       /* points into pool->params[id*(input_dim+1) + 1] */
    float       activation;     /* current activation value after applying activation function */
    float       error;          /* for GNG-style error accumulation (keep free for self-organizing use) */
    int         input_dim;      /* size of the weights vector */
} neuron_t;

typedef struct edge {
    int         from;           /* from the neuron id */
    int         to;             /* to the neuron id */
    int         active;         /* wether the connection is active */
    float       age;            /* wether the connection is new or old */
} edge_t;

/* Activate / initialize a pre-created neuron slot. */
void neuron_activate(neuron_t *n, activaton_t type, float *params, int input_dim);

/* Mark a neuron as inactive and clear its pointer state. */
void neuron_deactivate(neuron_t *n);

#endif
