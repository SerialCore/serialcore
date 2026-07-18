/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/layer.h>
#include <serialcore/sonn/activaton.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct layermeta {
    int start_id;
    int count;
    activaton_t act;
} layermeta_t;

/* Internal per-network layer state.
 * Layer is a special implementation built on top of the SONN core.
 * The core network_t knows nothing about layers.
 */
#define MAX_LAYERED_NETWORKS 8

typedef struct {
    network_t   *net;
    layermeta_t  layers[16];
    int          num_layers;
} layered_state_t;

static layered_state_t layered_states[MAX_LAYERED_NETWORKS];

static layered_state_t* get_layered_state(network_t *net)
{
    if (!net) return NULL;

    /* Find existing */
    for (int i = 0; i < MAX_LAYERED_NETWORKS; i++) {
        if (layered_states[i].net == net) {
            return &layered_states[i];
        }
    }

    /* Allocate new slot */
    for (int i = 0; i < MAX_LAYERED_NETWORKS; i++) {
        if (layered_states[i].net == NULL) {
            layered_states[i].net = net;
            layered_states[i].num_layers = 0;
            memset(layered_states[i].layers, 0, sizeof(layered_states[i].layers));
            return &layered_states[i];
        }
    }

    return NULL; /* too many */
}

static void reset_layered_state(layered_state_t *state)
{
    if (state) {
        state->num_layers = 0;
        memset(state->layers, 0, sizeof(state->layers));
    }
}

/* --- Internal helpers to reduce duplication --- */

static int get_layer_info(layered_state_t *state, int l,
                          int *start, int *cnt, activaton_t *act)
{
    if (!state || l < 0 || l >= state->num_layers) return -1;
    layermeta_t *m = &state->layers[l];
    if (start) *start = m->start_id;
    if (cnt)   *cnt   = m->count;
    if (act)   *act   = m->act;
    return 0;
}

static void register_layer(layered_state_t *state, int start, int count, activaton_t act)
{
    if (!state || state->num_layers >= 16) return;
    layermeta_t *m = &state->layers[state->num_layers++];
    m->start_id = start;
    m->count = count;
    m->act = act;
}

/* Accumulate weighted sum from previous layer neurons into *sum.
 * Uses the receiver's edge row + Option A (n->weights[s]).
 */
static void accumulate_from_prev_layer(nnpool_t *p, int nid,
                                       int prev_start, int prev_cnt,
                                       float *sum)
{
    if (!p || !sum) return;

    neuron_t *n = nnpool_get_neuron(p, nid);
    if (!n || !n->weights) return;

    edge_t *erow = nnpool_edge_row(p, nid);
    if (!erow) return;

    for (int s = 0; s < p->max_degree; s++) {
        if (!erow[s].active) continue;
        int from = erow[s].to;
        if (from >= prev_start && from < prev_start + prev_cnt) {
            neuron_t *prev = nnpool_get_neuron(p, from);
            if (prev) {
                *sum += prev->activation * n->weights[s];
            }
        }
    }
}

int layer_build_ffnn(network_t *net, const int *layer_sizes, int num_layers, const activaton_t *layer_acts)
{
    if (!net || !layer_sizes || num_layers < 2) return -1;

    layered_state_t *state = get_layered_state(net);
    if (!state) return -1;

    /* Build layers sequentially, fully connected to the next layer.
     * Layer is a special implementation built on the SONN core.
     * Core network_t knows nothing about layers.
     */
    reset_layered_state(state);

    int neuron_cursor = 0;

    /* Create input layer neurons (no incoming edges from previous layer) */
    activaton_t in_act = layer_acts ? layer_acts[0] : GELU;
    register_layer(state, neuron_cursor, layer_sizes[0], in_act);

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
        register_layer(state, layer_start, layer_sizes[l], act);

        int prev_start = 0, prev_cnt = 0;
        get_layer_info(state, l-1, &prev_start, &prev_cnt, NULL);

        for (int i = 0; i < layer_sizes[l]; i++) {
            int nid = network_add_neuron(net, act);
            if (nid < 0) return -1;

            /* Connect fully from previous layer */
            for (int j = 0; j < prev_cnt; j++) {
                int from = prev_start + j;
                /* Initial weight placed into the receiver (nid)'s weights[] at the
                 * local edge slot assigned for this connection (Option A).
                 */
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
    layered_state_t *state = get_layered_state(net);
    if (!state || layer_idx < 0 || layer_idx >= state->num_layers) return -1;
    if (start_id) *start_id = state->layers[layer_idx].start_id;
    if (count) *count = state->layers[layer_idx].count;
    return 0;
}

static float apply_act(float x, activaton_t a)
{
    return activaton_func(x, a);
}

static float apply_gradient(float x, activaton_t a)
{
    return gradient_func(x, a);
}

int layer_ffnn_forward(network_t *net, const float *inputs, float *outputs)
{
    if (!net || !inputs || !outputs) return -1;

    layered_state_t *state = get_layered_state(net);
    if (!state || state->num_layers < 2) return -1;

    nnpool_t *p = net->pool;
    if (!p) return -1;

    /* Store activations for this forward pass in the neuron's `activation` field.
     * `error` is reserved for self-organizing use (e.g. GNG error accumulation).
     */
    int in_start = 0, in_cnt = 0;
    get_layer_info(state, 0, &in_start, &in_cnt, NULL);

    for (int i = 0; i < in_cnt; i++) {
        neuron_t *n = nnpool_get_neuron(p, in_start + i);
        if (n) n->activation = inputs[i];
    }

    /* For each layer l >= 1 */
    for (int l = 1; l < state->num_layers; l++) {
        int start = 0, cnt = 0;
        activaton_t act = GELU;
        if (get_layer_info(state, l, &start, &cnt, &act) != 0) continue;

        int prev_start = 0, prev_cnt = 0;
        get_layer_info(state, l-1, &prev_start, &prev_cnt, NULL);

        for (int i = 0; i < cnt; i++) {
            int nid = start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (!n) continue;

            float sum = 0.0f;
            accumulate_from_prev_layer(p, nid, prev_start, prev_cnt, &sum);

            sum += n->bias;
            float out = apply_act(sum, act);
            n->activation = out; /* store activation for next layer */
        }
    }

    /* Copy final layer activations to outputs */
    int out_start = 0, out_cnt = 0;
    get_layer_info(state, state->num_layers-1, &out_start, &out_cnt, NULL);

    for (int i = 0; i < out_cnt; i++) {
        neuron_t *n = nnpool_get_neuron(p, out_start + i);
        outputs[i] = n ? n->activation : 0.0f;
    }

    return 0;
}

int layer_ffnn_train_step(network_t *net, const float *inputs, float target, float lr)
{
    if (!net || !inputs) return -1;

    layered_state_t *state = get_layered_state(net);
    if (!state || state->num_layers < 2) return -1;

    nnpool_t *p = net->pool;
    if (!p) return -1;

    float pred[1];
    if (layer_ffnn_forward(net, inputs, pred) != 0) return -1;

    float output_error = target - pred[0];

    /* === Backpropagation === */

    /* Zero deltas for all neurons in layers >=1 */
    for (int l = 1; l < state->num_layers; l++) {
        int start = 0, cnt = 0;
        get_layer_info(state, l, &start, &cnt, NULL);
        for (int i = 0; i < cnt; i++) {
            neuron_t *n = nnpool_get_neuron(p, start + i);
            if (n) n->delta = 0.0f;
        }
    }

    /* 1. Output layer deltas */
    int out_layer = state->num_layers - 1;
    int out_start = 0, out_cnt = 0;
    get_layer_info(state, out_layer, &out_start, &out_cnt, NULL);

    for (int i = 0; i < out_cnt; i++) {
        int nid = out_start + i;
        neuron_t *n = nnpool_get_neuron(p, nid);
        if (!n) continue;

        float act = n->activation;
        float grad = apply_gradient(act, n->type);
        /* For single-target output, apply the scalar error to each (current behavior) */
        n->delta = output_error * grad;
    }

    /* 2. Backpropagate deltas to hidden layers */
    for (int l = out_layer - 1; l >= 1; l--) {
        int curr_start = 0, curr_cnt = 0;
        get_layer_info(state, l, &curr_start, &curr_cnt, NULL);

        int next_start = 0, next_cnt = 0;
        get_layer_info(state, l + 1, &next_start, &next_cnt, NULL);

        /* Accumulate delta from next layer (reverse walk over next neuron's edges) */
        for (int j = 0; j < next_cnt; j++) {
            int next_nid = next_start + j;
            neuron_t *next_n = nnpool_get_neuron(p, next_nid);
            if (!next_n) continue;

            edge_t *erow = nnpool_edge_row(p, next_nid);
            if (!erow) continue;

            for (int s = 0; s < p->max_degree; s++) {
                if (!erow[s].active) continue;
                int from = erow[s].to;
                if (from >= curr_start && from < curr_start + curr_cnt) {
                    neuron_t *curr = nnpool_get_neuron(p, from);
                    if (curr) {
                        curr->delta += next_n->delta * next_n->weights[s];
                    }
                }
            }
        }

        /* Multiply by own derivative */
        for (int i = 0; i < curr_cnt; i++) {
            int nid = curr_start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (n) {
                float act = n->activation;
                float grad = apply_gradient(act, n->type);
                n->delta *= grad;
            }
        }
    }

    /* 3. Update weights and biases for all layers (from layer 1 to output) */
    for (int l = 1; l < state->num_layers; l++) {
        int start = 0, cnt = 0, prev_start = 0, prev_cnt = 0;
        get_layer_info(state, l, &start, &cnt, NULL);
        get_layer_info(state, l-1, &prev_start, &prev_cnt, NULL);

        for (int i = 0; i < cnt; i++) {
            int nid = start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (!n) continue;

            /* Use similar walk as forward but apply gradient */
            edge_t *erow = nnpool_edge_row(p, nid);
            if (!erow) continue;

            for (int s = 0; s < p->max_degree; s++) {
                if (!erow[s].active) continue;
                int from = erow[s].to;
                if (from >= prev_start && from < prev_start + prev_cnt) {
                    neuron_t *prev = nnpool_get_neuron(p, from);
                    float prev_act = prev ? prev->activation : 0.0f;

                    n->weights[s] += lr * n->delta * prev_act;
                }
            }

            /* Update bias */
            n->bias += lr * n->delta;
        }
    }

    return 0;
}
