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
    if (state->num_layers < 16) {
        state->layers[state->num_layers].start_id = neuron_cursor;
        state->layers[state->num_layers].count = layer_sizes[0];
        state->layers[state->num_layers].act = in_act;
        state->num_layers++;
    }
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
        if (state->num_layers < 16) {
            state->layers[state->num_layers].start_id = layer_start;
            state->layers[state->num_layers].count = layer_sizes[l];
            state->layers[state->num_layers].act = act;
            state->num_layers++;
        }

        for (int i = 0; i < layer_sizes[l]; i++) {
            int nid = network_add_neuron(net, act);
            if (nid < 0) return -1;

            /* Connect fully from previous layer */
            int prev_start = state->layers[l-1].start_id;
            int prev_cnt = state->layers[l-1].count;

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
    extern float activaton_func(float x, activaton_t a);
    return activaton_func(x, a);
}

static float apply_gradient(float x, activaton_t a)
{
    extern float gradient_func(float x, activaton_t a);
    return gradient_func(x, a);
}

int layer_ffnn_forward(network_t *net, const float *inputs, float *outputs)
{
    if (!net || !inputs || !outputs) return -1;

    layered_state_t *state = get_layered_state(net);
    if (!state || state->num_layers < 2) return -1;

    nnpool_t *p = network_get_pool(net);
    if (!p) return -1;

    /* Store activations for this forward pass in the neuron's `activation` field.
     * `error` is reserved for self-organizing use (e.g. GNG error accumulation).
     */
    int in_start = state->layers[0].start_id;
    int in_cnt = state->layers[0].count;

    for (int i = 0; i < in_cnt; i++) {
        neuron_t *n = nnpool_get_neuron(p, in_start + i);
        if (n) n->activation = inputs[i];
    }

    /* For each layer l >= 1 */
    for (int l = 1; l < state->num_layers; l++) {
        int start = state->layers[l].start_id;
        int cnt = state->layers[l].count;
        activaton_t act = state->layers[l].act;

        for (int i = 0; i < cnt; i++) {
            int nid = start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (!n) continue;

            float sum = 0.0f;

            /* Sum over previous layer */
            int prev_start = state->layers[l-1].start_id;
            int prev_cnt = state->layers[l-1].count;

            edge_t *erow = nnpool_edge_row(p, nid);
            if (!erow) continue;

            for (int s = 0; s < p->max_degree; s++) {
                if (!erow[s].active) continue;
                int from = erow[s].to;

                /* Only consider neurons from previous layer.
                 * Weight for this incoming connection is stored at the receiver's
                 * weights[s] (Option A: local edge slot index == weight index).
                 */
                if (from >= prev_start && from < prev_start + prev_cnt) {
                    neuron_t *prev = nnpool_get_neuron(p, from);
                    if (prev) {
                        float w = n->weights[s];   /* use neuron.weights directly */
                        sum += prev->activation * w;
                    }
                }
            }

            sum += n->bias;
            float out = apply_act(sum, act);
            n->activation = out; /* store activation for next layer */
        }
    }

    /* Copy final layer activations to outputs */
    int out_start = state->layers[state->num_layers-1].start_id;
    int out_cnt = state->layers[state->num_layers-1].count;

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

    nnpool_t *p = network_get_pool(net);
    if (!p) return -1;

    float pred[1];
    if (layer_ffnn_forward(net, inputs, pred) != 0) return -1;

    float output_error = target - pred[0];

    /* === Backpropagation === */

    /* Zero deltas for all neurons in layers >=1 */
    for (int l = 1; l < state->num_layers; l++) {
        int start = state->layers[l].start_id;
        int cnt = state->layers[l].count;
        for (int i = 0; i < cnt; i++) {
            neuron_t *n = nnpool_get_neuron(p, start + i);
            if (n) n->delta = 0.0f;
        }
    }

    /* 1. Output layer deltas */
    int out_layer = state->num_layers - 1;
    int out_start = state->layers[out_layer].start_id;
    int out_cnt = state->layers[out_layer].count;

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
        int curr_start = state->layers[l].start_id;
        int curr_cnt = state->layers[l].count;
        int next_start = state->layers[l + 1].start_id;
        int next_cnt = state->layers[l + 1].count;

        /* Accumulate from next layer */
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
                    float w = next_n->weights[s];
                    neuron_t *curr = nnpool_get_neuron(p, from);
                    if (curr) {
                        curr->delta += next_n->delta * w;
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
        int start = state->layers[l].start_id;
        int cnt = state->layers[l].count;
        int prev_start = state->layers[l - 1].start_id;
        int prev_cnt = state->layers[l - 1].count;

        for (int i = 0; i < cnt; i++) {
            int nid = start + i;
            neuron_t *n = nnpool_get_neuron(p, nid);
            if (!n) continue;

            edge_t *erow = nnpool_edge_row(p, nid);
            if (!erow) continue;

            for (int s = 0; s < p->max_degree; s++) {
                if (!erow[s].active) continue;
                int from = erow[s].to;
                if (from < prev_start || from >= prev_start + prev_cnt) continue;

                neuron_t *prev = nnpool_get_neuron(p, from);
                float prev_act = prev ? prev->activation : 0.0f;

                float w = n->weights[s];
                w += lr * n->delta * prev_act;
                n->weights[s] = w;
            }

            /* Update bias */
            n->bias += lr * n->delta;
        }
    }

    return 0;
}
