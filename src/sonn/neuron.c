/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/neuron.h>

#include <stdlib.h>
#include <time.h>

void neuron_activate(neuron_t *n, activaton_t type, float *weights, int weight_count)
{
    if (!n) return;

    /* ID is pre-assigned by Pool during pool creation.
     * Do not overwrite it here. Network has no permission to touch neuron IDs.
     */
    n->type = type;
    n->error = 0.0f;
    n->active = 1;
    n->weights = weights;

    srand(time(NULL));
    /* All initializations embedded here: random for weights and bias */
    if (weights && weight_count > 0) {
        for (int i = 0; i < weight_count; i++) {
            float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            weights[i] = r * 0.1f;
        }
    }

    float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    n->bias = r * 0.01f;
}

void neuron_deactivate(neuron_t *n)
{
    if (!n) return;

    n->active = 0;
    n->bias = 0.0f;
    n->weights = NULL;
}