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
    n->bias = params ? params[0] : 0.0f;
    n->weights = params ? params + 1 : NULL;
    n->input_dim = input_dim;
}

void neuron_deactivate(neuron_t *n)
{
    if (!n) return;

    n->active = 0;
    n->bias = 0.0f;
    n->weights = NULL;
}
