/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/ffnn.h>
#include <serialcore/ffnn/dense.h>
#include <serialcore/ffnn/gemm.h>
#include <serialcore/math/xoshiross.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Map 53 high bits of next() into [0,1). Used by the He-init path for
 * the Dense layer family. */
static float generate_random_parameter(void)
{
    uint64_t r = next();
    return (float)((r >> 11) * (1.0 / 9007199254740992.0));
}

ffnn_network_t* ffnn_create(int inputs, int batch, float learning_rate, float momentum, float decay)
{
    if (inputs <= 0 || batch <= 0) return NULL;

    ffnn_network_t *net = (ffnn_network_t *)calloc(1, sizeof(*net));
    if (!net) return NULL;

    net->cap = 8;
    net->layers = (ffnn_layer_t *)calloc(net->cap, sizeof(ffnn_layer_t));
    if (!net->layers) { free(net); return NULL; }

    net->n = 0;
    net->inputs = inputs;
    net->outputs = inputs;
    net->batch  = batch;
    net->learning_rate = learning_rate;
    net->momentum = momentum;
    net->decay = decay;
    net->train = 0;
    net->index = 0;

    net->input_buffer = (float *)calloc((size_t)batch * inputs, sizeof(float));
    net->input = net->input_buffer;
    net->delta = NULL;
    return net;
}

void ffnn_destroy(ffnn_network_t *net)
{
    if (!net) return;
    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        free(l->weights);
        free(l->biases);
        free(l->weight_updates);
        free(l->bias_updates);
        free(l->output);
        free(l->pre_act);
        free(l->delta);
        free(l->input_snapshot);
    }
    free(net->layers);
    free(net->input_buffer);
    free(net);
}

/* Build a layer of `type` in `*l` (which must already be zeroed). Today
 * only FFNN_DENSE is wired up; the other families are reserved slots that
 * record `type`/shape so the bookkeeping in ffnn_add_layer stays
 * consistent. Returns 0 on success, -1 on an unknown/unsupported type. */
static int ffnn_build_layer(ffnn_layer_t *l, int batch, int inputs, int outputs, layertype_t type, activaton_t activation)
{
    switch (type) {
    case FFNN_DENSE: {
        l->type       = FFNN_DENSE;
        l->activation = activation;
        l->inputs     = inputs;
        l->outputs    = outputs;
        l->batch      = batch;

        l->weights        = (float *)calloc((size_t)outputs * inputs, sizeof(float));
        l->biases         = (float *)calloc((size_t)outputs, sizeof(float));
        l->weight_updates = (float *)calloc((size_t)outputs * inputs, sizeof(float));
        l->bias_updates   = (float *)calloc((size_t)outputs, sizeof(float));

        l->output         = (float *)calloc((size_t)batch * outputs, sizeof(float));
        l->pre_act        = (float *)calloc((size_t)batch * outputs, sizeof(float));
        l->delta          = (float *)calloc((size_t)batch * outputs, sizeof(float));
        l->input_snapshot = (float *)calloc((size_t)batch * inputs, sizeof(float));

        if (!l->weights || !l->biases || !l->weight_updates || !l->bias_updates ||
            !l->output  || !l->pre_act || !l->delta || !l->input_snapshot) {
            return -1;     /* ffnn_destroy will free what was allocated */
        }

        /* He-init for ReLU-family / GELU:
         *     scale = sqrt(2.0 / inputs);  w ~ scale * uniform(-1, 1) */
        const float scale = sqrtf(2.0f / (float)inputs);
        for (int i = 0; i < outputs * inputs; ++i) {
            l->weights[i] = scale * (2.0f * generate_random_parameter() - 1.0f);
        }
        for (int o = 0; o < outputs; ++o) {
            l->biases[o] = 0.0f;
        }

        l->forward  = dense_forward;
        l->backward = dense_backward;
        l->update   = dense_update;
        return 0;
    }

    case FFNN_CONVOLUTIONAL:
    case FFNN_MAXPOOL:
    case FFNN_BATCHNORM:
    case FFNN_SOFTMAX:
    case FFNN_BLANK:
    default:
        l->type       = type;
        l->activation = activation;
        l->inputs     = inputs;
        l->outputs    = outputs;
        l->batch      = batch;
        return 0;
    }
}

int ffnn_add_layer(ffnn_network_t *net, int inputs, int outputs, layertype_t type, activaton_t activation)
{
    if (!net) return -1;

    const int expect = (net->n == 0) ? net->inputs : net->layers[net->n - 1].outputs;
    if (inputs != expect) {
        fprintf(stderr,
                "ffnn: layer %d inputs=%d != expected %d\n",
                net->n, inputs, expect);
        return -1;
    }

    if (net->n == net->cap) {
        int new_cap = net->cap * 2;
        ffnn_layer_t *p = (ffnn_layer_t *)realloc(net->layers, (size_t)new_cap * sizeof(ffnn_layer_t));
        if (!p) return -1;
        net->layers = p;
        net->cap = new_cap;
    }

    ffnn_layer_t *slot = &net->layers[net->n];
    *slot = (ffnn_layer_t){0};

    if (ffnn_build_layer(slot, net->batch, inputs, outputs, type, activation) != 0) {
        /* Free anything the partial build allocated; the slot is now
         * zeroed so ffnn_destroy won't double-free later. */
        free(slot->weights);        slot->weights = NULL;
        free(slot->biases);         slot->biases = NULL;
        free(slot->weight_updates); slot->weight_updates = NULL;
        free(slot->bias_updates);   slot->bias_updates = NULL;
        free(slot->output);         slot->output = NULL;
        free(slot->pre_act);        slot->pre_act = NULL;
        free(slot->delta);          slot->delta = NULL;
        free(slot->input_snapshot); slot->input_snapshot = NULL;
        return -1;
    }

    net->n++;
    net->outputs = outputs;
    return 0;
}

void ffnn_forward(ffnn_network_t *net, const float *input, float *output)
{
    if (!net || !input || !output) return;

    /* Reset net->input to the owned buffer so the caller's data lands in a
     * stable place even after the previous forward repointed it. */
    net->input = net->input_buffer;
    for (int i = 0; i < net->batch * net->inputs; ++i) {
        net->input[i] = input[i];
    }

    net->train = 1;            /* cache inputs during forward for backward */

    for (int i = 0; i < net->n; ++i) {
        net->index = i;
        ffnn_layer_t *l = &net->layers[i];

        if (l->delta) {
            gemm_fill(l->batch * l->outputs, 0.0f, l->delta, 1);
        }
        if (l->forward) {
            l->forward(l, net);
        }
        net->input = l->output;     /* feed the next layer */
    }

    /* Read out the last layer's activation. */
    ffnn_layer_t *last = &net->layers[net->n - 1];
    for (int i = 0; i < net->batch * net->outputs; ++i) {
        output[i] = last->output[i];
    }
}

void ffnn_backward(ffnn_network_t *net, const float *target)
{
    if (!net || !target) return;

    /* Restart from the output layer. For MSE loss L = 0.5 * (y - t)^2 the
     *  descent-compatible delta is (t - y); weight_updates then accumulate
     *  `delta * input` and `weights += lr/batch * weight_updates` performs the
     *  descent step (matches darknet's l2_cpu convention). */
    ffnn_layer_t *last = &net->layers[net->n - 1];
    for (int i = 0; i < net->batch * net->outputs; ++i) {
        last->delta[i] = (target[i] - last->output[i]);
    }

    /* Walk layers in reverse. Before each backward call we set
     * net->input  = prev.output   (the input this layer saw during forward)
     * net->delta  = prev.delta     (where this layer deposits the input-g) */
    for (int i = net->n - 1; i >= 0; --i) {
        ffnn_layer_t *l = &net->layers[i];
        ffnn_layer_t *prev = (i == 0) ? NULL : &net->layers[i - 1];

        if (prev) {
            net->input = prev->output;
            net->delta = prev->delta;
            if (prev->delta) {
                gemm_fill(prev->batch * prev->outputs, 0.0f, prev->delta, 1);
            }
        } else {
            net->delta = NULL;
        }
        net->index = i;
        if (l->backward) {
            l->backward(l, net);
        }
    }
}

void ffnn_update(ffnn_network_t *net)
{
    if (!net) return;
    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        if (l->update) {
            l->update(l, net->learning_rate, net->momentum, net->decay, net->batch);
        }
    }
}

void ffnn_train_step(ffnn_network_t *net, const float *input, const float *target)
{
    if (!net || !input || !target) return;

    float *out = (float *)malloc((size_t)net->batch * net->outputs * sizeof(float));
    if (!out) return;

    ffnn_forward(net, input, out);
    ffnn_backward(net, target);
    ffnn_update(net);

    free(out);
}

void ffnn_predict(ffnn_network_t *net, const float *input, float *output)
{
    if (!net) return;
    int saved_train = net->train;
    net->train = 0;             /* don't cache snapshots for inference */
    ffnn_forward(net, input, output);
    net->train = saved_train;
}