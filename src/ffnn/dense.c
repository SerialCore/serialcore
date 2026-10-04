/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/dense.h>
#include <serialcore/ffnn/gemm.h>
#include <serialcore/math/xoshiross.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct dense_state {
    float *weights;          /* pool params, not owned */
    float *biases;
    float *weight_updates;   /* pool grads, not owned */
    float *bias_updates;
    float *output;           /* [batch * outputs] */
    float *pre_act;
    float *delta;
    float *input_snapshot;   /* [batch * inputs] */
} dense_state_t;

static dense_state_t *dense_of(ffnn_layer_t *l)
{
    return l ? (dense_state_t *)l->state : NULL;
}

static int dense_shape(const ffnn_layer_t *l)
{
    if (!l || l->inputs <= 0 || l->outputs <= 0 || l->batch <= 0) return -1;
    return 0;
}

static int dense_count_params(const ffnn_layer_t *l)
{
    long long n;
    if (dense_shape(l) != 0) return -1;
    if (l->outputs > INT_MAX / l->inputs) return -1;
    n = (long long)l->outputs * l->inputs + l->outputs;
    if (n > INT_MAX) return -1;
    return (int)n;
}

static int dense_bind(ffnn_layer_t *l, float *params, float *grads)
{
    dense_state_t *st = dense_of(l);
    int nw;
    if (!st || !params || !grads || dense_count_params(l) < 0) return -1;
    nw = l->outputs * l->inputs;
    st->weights = params;
    st->biases = params + nw;
    st->weight_updates = grads;
    st->bias_updates = grads + nw;
    return 0;
}

static float dense_uniform_sym(void)
{
    uint64_t r = next();
    float u = (float)((r >> 11) * (1.0 / 9007199254740992.0));
    return 2.0f * u - 1.0f;
}

static void dense_init(ffnn_layer_t *l)
{
    dense_state_t *st = dense_of(l);
    int nw;
    float limit;

    if (!st || !st->weights || !st->biases) return;
    nw = l->outputs * l->inputs;

    /* ReLU-family, including GELU and Identity: U[-a, a] with a = sqrt(2/fan_in).
     * Saturating units use Xavier uniform, a = sqrt(6/(fan_in + fan_out)).
     * The wider He uniform a = sqrt(6/fan_in) stalls the XOR network at lr 0.1. */
    switch (l->activation) {
        case Sigmoid:
        case Tanh:
        case Sin:
        case Cos:
            limit = sqrtf(6.0f / (float)(l->inputs + l->outputs));
            break;
        default:
            limit = sqrtf(2.0f / (float)l->inputs);
            break;
    }
    for (int i = 0; i < nw; ++i) st->weights[i] = dense_uniform_sym() * limit;
    for (int o = 0; o < l->outputs; ++o) st->biases[o] = 0.0f;
}

static int dense_forward(ffnn_layer_t *l, ffnn_network_t *net)
{
    dense_state_t *st = dense_of(l);
    int nout;

    if (!st || !net || !net->input || !st->weights || !st->output || !st->pre_act) return -1;

    gemm(0, 1, l->batch, l->outputs, l->inputs,
         1.0f, net->input, l->inputs,
         st->weights, l->inputs,
         0.0f, st->pre_act, l->outputs);
    gemm_add_bias(st->biases, st->pre_act, l->batch, l->outputs);

    nout = l->batch * l->outputs;
    for (int i = 0; i < nout; ++i) st->output[i] = st->pre_act[i];
    gemm_activate_array(st->output, nout, l->activation);

    if (net->train && st->input_snapshot) {
        int nin = l->batch * l->inputs;
        for (int i = 0; i < nin; ++i) st->input_snapshot[i] = net->input[i];
    }
    return 0;
}

static int dense_backward(ffnn_layer_t *l, ffnn_network_t *net)
{
    dense_state_t *st = dense_of(l);
    if (!st || !st->delta || !st->pre_act || !st->weights || !st->input_snapshot) return -1;

    gemm_gradient_array(st->pre_act, l->batch * l->outputs, l->activation, st->delta);
    gemm_backward_bias(st->bias_updates, st->delta, l->batch, l->outputs);
    gemm(1, 0, l->outputs, l->inputs, l->batch,
         1.0f, st->delta, l->outputs,
         st->input_snapshot, l->inputs,
         1.0f, st->weight_updates, l->inputs);

    if (net && net->delta) {
        gemm(0, 0, l->batch, l->inputs, l->outputs,
             1.0f, st->delta, l->outputs,
             st->weights, l->inputs,
             1.0f, net->delta, l->inputs);
    }
    return 0;
}

static void dense_update(ffnn_layer_t *l, float lr, float momentum, float decay, int batch)
{
    dense_state_t *st = dense_of(l);
    int nw;
    if (!st || !st->weights || !st->biases) return;
    nw = l->outputs * l->inputs;
    /* Biases are not decayed. Weights use w += (lr/batch) * (update - decay * batch * w). */
    ffnn_sgd_step(st->biases, st->bias_updates, l->outputs, lr, momentum, decay, batch, 0);
    ffnn_sgd_step(st->weights, st->weight_updates, nw, lr, momentum, decay, batch, 1);
}

static float *dense_output(ffnn_layer_t *l)
{
    dense_state_t *st = dense_of(l);
    return st ? st->output : NULL;
}

static float *dense_delta(ffnn_layer_t *l)
{
    dense_state_t *st = dense_of(l);
    return st ? st->delta : NULL;
}

static void dense_release(ffnn_layer_t *l)
{
    dense_state_t *st = dense_of(l);
    if (!st) return;
    free(st->output);
    free(st->pre_act);
    free(st->delta);
    free(st->input_snapshot);
    free(st);
    l->state = NULL;
}

static const ffnn_layer_ops_t DENSE_OPS = {
    FFNN_DENSE,
    dense_shape,
    dense_count_params,
    dense_bind,
    dense_init,
    dense_forward,
    dense_backward,
    dense_update,
    dense_output,
    dense_delta,
    NULL,
    NULL,
    dense_release
};

const ffnn_layer_ops_t *dense_layer_ops(void)
{
    return &DENSE_OPS;
}

int dense_prepare(ffnn_layer_t *l)
{
    dense_state_t *st;
    size_t nout, nin;

    if (!l || dense_shape(l) != 0) return -1;
    st = (dense_state_t *)calloc(1, sizeof(*st));
    if (!st) return -1;

    nout = (size_t)l->batch * (size_t)l->outputs;
    nin = (size_t)l->batch * (size_t)l->inputs;
    st->output = (float *)calloc(nout, sizeof(float));
    st->pre_act = (float *)calloc(nout, sizeof(float));
    st->delta = (float *)calloc(nout, sizeof(float));
    st->input_snapshot = (float *)calloc(nin, sizeof(float));
    l->state = st;
    l->ops = dense_layer_ops();
    if (!st->output || !st->pre_act || !st->delta || !st->input_snapshot) {
        dense_release(l);
        return -1;
    }
    return 0;
}
