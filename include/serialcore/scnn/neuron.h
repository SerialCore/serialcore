/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SCNN_NEURON
#define SERIALCORE_SCNN_NEURON

#include <serialcore/scnn/activation.h>

/* define the number of synapses to be 1000 according to biological study */
#define SYNAPSE 1000

typedef struct neuron {
    int id;
    ACTIVATION_T type;

    float bias;
    float weight[SYNAPSE];

    int *forward;
    int *backward[SYNAPSE];
}neuron_t;

#endif
