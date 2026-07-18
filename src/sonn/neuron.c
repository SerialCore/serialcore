/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/neuron.h>

#include <stdlib.h>

void neuron_activate(neuron_t *n, activaton_t type, float *params, int input_dim)
{
    if (!n) return;

    /* ID is pre-assigned by Pool during pool creation.
     * Do not overwrite it here. Network has no permission to touch neuron IDs.
     * Unified parameter layout: bias at params[0], weights at params[1..].
     */
    n->active = 1;
    n->type = type;
    n->error = 0.0f;
    n->activation = 0.0f;
    n->delta = 0.0f;
    n->weights = params ? params + 1 : NULL;
    n->input_dim = input_dim;

    if (params) {
        n->bias = params[0];
    }
}

void neuron_deactivate(neuron_t *n)
{
    if (!n) return;

    n->active = 0;
    n->activation = 0.0f;
    n->delta = 0.0f;
    n->weights = NULL;
}