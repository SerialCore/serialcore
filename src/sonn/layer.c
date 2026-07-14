/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/layer.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct layermeta {
    int start_id;
    int count;
    activaton_t act;
} layermeta_t;

/* We store layer metadata in a very small side structure attached
 * via a global table keyed by the Network pointer (simple, one-network demo).
 * For real code this would be stored inside Network or in a wrapper.
 */
static layermeta_t g_layers[16];
static int g_num_layers = 0;
static network_t *g_last_net = NULL;

static void reset_layers(void) {
    g_num_layers = 0;
    memset(g_layers, 0, sizeof(g_layers));
}

static void record_layer(int start_id, int count, activaton_t act) {
    if (g_num_layers >= 16) return;
    g_layers[g_num_layers].start_id = start_id;
    g_layers[g_num_layers].count = count;
    g_layers[g_num_layers].act = act;
    g_num_layers++;
}

int layer_build_ffnn(network_t *net,
                          const int *layer_sizes,
                          int num_layers,
                          const activaton_t *layer_acts)
{
    if (!net || !layer_sizes || num_layers < 2) return -1;

    /* Build layers sequentially, fully connected to the next layer. */
    reset_layers();
    g_last_net = net;

    int neuron_cursor = 0;

    /* Create input layer neurons (no incoming edges from previous layer) */
    activaton_t in_act = layer_acts ? layer_acts[0] : GELU;
    record_layer(neuron_cursor, layer_sizes[0], in_act);
    for (int i = 0; i < layer_sizes[0]; i++) {
        int nid = network_add_neuron(net, in_act);
        if (nid != neuron_cursor) {
            /* We assume sequential ids starting from 0 */
        }
        neuron_cursor++;
    }

    /* For each subsequent layer */
    for (int l = 1; l < num_layers; l++) {
        activaton_t act = layer_acts ? layer_acts[l] : GELU;
        int layer_start = neuron_cursor;
        record_layer(layer_start, layer_sizes[l], act);

        for (int i = 0; i < layer_sizes[l]; i++) {
            int nid = network_add_neuron(net, act);
            if (nid < 0) return -1;

            /* Connect fully from previous layer */
            int prev_start = g_layers[l-1].start_id;
            int prev_cnt = g_layers[l-1].count;

            for (int j = 0; j < prev_cnt; j++) {
                int from = prev_start + j;
                /* Random-ish small weights on the *edges* (used by current forward) */
                float w = 0.1f * ((float)((nid + j) % 7) - 3.0f);
                network_add_edge(net, from, nid, w);
            }

            neuron_cursor++;
        }
    }

    /* Store some config in the Network struct for convenience */
    net->input_dim = layer_sizes[0];
    /* We don't have direct max_neurons in network anymore, it's in Pool. */

    return 0;
}

int layer_get_range(network_t *net, int layer_idx, int *start_id, int *count)
{
    if (!net || layer_idx < 0 || layer_idx >= g_num_layers) return -1;
    if (start_id) *start_id = g_layers[layer_idx].start_id;
    if (count) *count = g_layers[layer_idx].count;
    return 0;
}

static float apply_act(float x, activaton_t a) {
    extern float activaton_func(float x, activaton_t a);
    return activaton_func(x, a);
}

int layer_ffnn_forward(network_t *net, const float *inputs, float *outputs)
{
    if (!net || !inputs || !outputs) return -1;
    if (g_last_net != net || g_num_layers < 2) return -1;

    nnpool_t *p = network_get_pool(net);
    if (!p) return -1;

    /* Store activations for this forward pass in the neuron's `error` field
     * (a convenient existing float; not used for its normal purpose here).
     */
    int in_start = g_layers[0].start_id;
    int in_cnt = g_layers[0].count;

    for (int i = 0; i < in_cnt; i++) {
        neuron_t *n = nnpool_get_neuron(p, in_start + i);
        if (n) n->error = inputs[i];
    }

    /* For each layer l >= 1 */
    for (int l = 1; l < g_num_layers; l++) {
        int start = g_layers[l].start_id;
        int cnt = g_layers[l].count;
        activaton_t act = g_layers[l].act;

        for (int i = 0; i < cnt; i++) {
            int nid = start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (!n) continue;

            float sum = 0.0f;

            /* Sum over previous layer */
            int prev_start = g_layers[l-1].start_id;
            int prev_cnt = g_layers[l-1].count;

            int *row = nnpool_adjacency_row(p, nid);
            if (!row) continue;

            for (int s = 0; s < p->max_degree; s++) {
                int from = row[s];
                if (from < 0) continue;

                /* Only consider neurons from previous layer */
                if (from >= prev_start && from < prev_start + prev_cnt) {
                    neuron_t *prev = nnpool_get_neuron(p, from);
                    if (prev) {
                        /* Find edge weight */
                        int slot = nnpool_find_edge_slot(p, from, nid);
                        float w = 0.0f;
                        if (slot >= 0) {
                            int eidx = from * p->max_degree + slot;
                            w = p->edges[eidx].strength;
                        }
                        sum += prev->error * w;
                    }
                }
            }

            sum += n->bias;
            float out = apply_act(sum, act);
            n->error = out; /* store activation for next layer using error field */
        }
    }

    /* Copy final layer activations to outputs */
    int out_start = g_layers[g_num_layers-1].start_id;
    int out_cnt = g_layers[g_num_layers-1].count;

    for (int i = 0; i < out_cnt; i++) {
        neuron_t *n = nnpool_get_neuron(p, out_start + i);
        outputs[i] = n ? n->error : 0.0f;
    }

    return 0;
}

int layer_ffnn_train_step(network_t *net, const float *inputs, float target, float lr)
{
    if (!net || !inputs) return -1;
    if (g_num_layers < 2) return -1;

    nnpool_t *p = network_get_pool(net);
    if (!p) return -1;

    float pred[1];
    if (layer_ffnn_forward(net, inputs, pred) != 0) return -1;

    float error = target - pred[0];

    /* Update weights into the output layer */
    int out_layer = g_num_layers - 1;
    int out_start = g_layers[out_layer].start_id;
    int out_cnt = g_layers[out_layer].count;

    int prev_start = g_layers[out_layer-1].start_id;
    int prev_cnt = g_layers[out_layer-1].count;

    /* For each output neuron */
    for (int i = 0; i < out_cnt; i++) {
        int nid = out_start + i;
        neuron_t *n = nnpool_get_neuron(p, nid);
        if (!n) continue;

        int *row = nnpool_adjacency_row(p, nid);
        if (!row) continue;

        for (int s = 0; s < p->max_degree; s++) {
            int from = row[s];
            if (from < 0) continue;
            if (from < prev_start || from >= prev_start + prev_cnt) continue;

            int slot = nnpool_find_edge_slot(p, from, nid);
            if (slot < 0) continue;

            int eidx = from * p->max_degree + slot;
            float w = p->edges[eidx].strength;

            /* Get previous activation (stored in error field) */
            neuron_t *prev = nnpool_get_neuron(p, from);
            float prev_act = prev ? prev->error : 0.0f;

            /* Delta rule: w += lr * error * prev_act */
            w += lr * error * prev_act;
            p->edges[eidx].strength = w;
        }

        /* Update bias for this output neuron (the "input" to bias is always 1) */
        n->bias += lr * error;
    }

    return 0;
}
