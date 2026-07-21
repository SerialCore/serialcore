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

    n->active = 1;
    n->type = type;
    n->weights = params ? params + 1 : NULL;
    n->activation = 0.0f;
    n->error = 0.0f;
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
    n->error = 0.0f;
    n->weights = NULL;
}