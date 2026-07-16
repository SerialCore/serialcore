/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/neuron.h>

#include <stdlib.h>
#include <time.h>

void neuron_activate(neuron_t *n, activaton_t type, float *params, int input_dim)
{
    if (!n) return;

    /* ID is pre-assigned by Pool during pool creation.
     * Do not overwrite it here. Network has no permission to touch neuron IDs.
     */
    n->active = 1;
    n->type = type;
    n->error = 0.0f;
    n->weights = params ? params + 1 : NULL;
    n->input_dim = input_dim;

    srand(time(NULL));
    /* Unified parameter layout: bias at params[0], weights at params[1..] */
    if (params) {
        /* bias (index 0) */
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        params[0] = r * 0.01f;
        n->bias = params[0];

        /* weights */
        if (input_dim > 0) {
            for (int i = 0; i < input_dim; i++) {
                float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                params[i + 1] = r * 0.1f;
            }
        }
    }
}

void neuron_deactivate(neuron_t *n)
{
    if (!n) return;

    n->active = 0;
    n->weights = NULL;
}