/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/ffnn.h>
#include <serialcore/ffnn/dense.h>
#include <serialcore/ffnn/gemm.h>
#include <serialcore/ffnn/mmpool.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    net->compiled = 0;
    net->pool = NULL;

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
        free(l->output);
        free(l->pre_act);
        free(l->delta);
        free(l->input_snapshot);
    }
    free(net->layers);
    free(net->input_buffer);
    mmpool_destroy(net->pool);
    free(net);
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

    int batch = net->batch;
    int layer_built = 0;
    switch (type) {
        case FFNN_DENSE: {
            slot->type       = FFNN_DENSE;
            slot->activation = activation;
            slot->inputs     = inputs;
            slot->outputs    = outputs;
            slot->batch      = batch;

            /* Learnable parameters are NOT allocated here. They live in net->pool after
             * ffnn_compile() runs. Until then these pointers stay NULL; the
             * test harness must call ffnn_compile() before any forward pass. */
            slot->weights        = NULL;
            slot->biases         = NULL;
            slot->weight_updates = NULL;
            slot->bias_updates   = NULL;

            /* Per-batch workspace belongs to the layer (it depends on `batch`
             * and is never serialized, so it has no place in the pool). */
            slot->output         = (float *)calloc((size_t)batch * outputs, sizeof(float));
            slot->pre_act        = (float *)calloc((size_t)batch * outputs, sizeof(float));
            slot->delta          = (float *)calloc((size_t)batch * outputs, sizeof(float));
            slot->input_snapshot = (float *)calloc((size_t)batch * inputs, sizeof(float));

            if (!slot->output || !slot->pre_act || !slot->delta || !slot->input_snapshot) {
                layer_built = -1;     /* ffnn_destroy will free what was allocated */
            }

            slot->forward  = dense_forward;
            slot->backward = dense_backward;
            slot->update   = dense_update;
            layer_built = 0;
        }

        case FFNN_CONVOLUTIONAL:
        case FFNN_MAXPOOL:
        case FFNN_BATCHNORM:
        case FFNN_SOFTMAX:
        case FFNN_BLANK:
        default:
            slot->type       = type;
            slot->activation = activation;
            slot->inputs     = inputs;
            slot->outputs    = outputs;
            slot->batch      = batch;
            layer_built = 0;
    }

    if (layer_built != 0) {
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

int ffnn_compile(ffnn_network_t *net)
{
    if (!net || net->n <= 0) return -1;
    if (net->compiled) return 0;        /* idempotent */

    /* Materialize the per-layer inputs[]/outputs[] arrays the pool expects. */
    int *inputs_arr  = (int*)calloc((size_t)net->n, sizeof(int));
    int *outputs_arr = (int*)calloc((size_t)net->n, sizeof(int));
    if (!inputs_arr || !outputs_arr) {
        free(inputs_arr); free(outputs_arr);
        return -1;
    }
    for (int i = 0; i < net->n; ++i) {
        inputs_arr[i]  = net->layers[i].inputs;
        outputs_arr[i] = net->layers[i].outputs;
    }

    net->pool = mmpool_create(inputs_arr, outputs_arr, net->n);
    free(inputs_arr);
    free(outputs_arr);
    if (!net->pool) return -1;

    /* Wire per-layer views into the pool's contiguous arenas. The GEMM
     * kernels in dense.c keep addressing l->weights, l->biases, etc. —
     * they don't care that the storage is now a sub-pointer into one big
     * arena rather than a standalone allocation. */
    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        l->weights        = mmpool_get_weights(net->pool, i);
        l->biases         = mmpool_get_biases(net->pool, i);
        l->weight_updates = mmpool_get_weight_updates(net->pool, i);
        l->bias_updates   = mmpool_get_bias_updates(net->pool, i);
    }

    net->compiled = 1;
    return 0;
}

void ffnn_forward(ffnn_network_t *net, const float *input, float *output)
{
    if (!net || !input || !output) return;
    if (!net->compiled || !net->pool) {
        fprintf(stderr, "ffnn_forward: network not compiled — call ffnn_compile()\n");
        return;
    }

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