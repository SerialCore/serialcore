/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/sonn.h>

#include <stdlib.h>
#include <string.h>

static int find_free_edge_slot(edge_t *row, int md)
{
    if (!row) return -1;
    for (int s = 0; s < md; s++) {
        if (!row[s].active) return s;
    }
    return -1;
}

static void clear_weight_slot(nnpool_t *pool, int neuron, int slot)
{
    float *w = nnpool_get_params(pool, neuron);
    if (w) w[slot + 1] = 0.0f;
}

/* Add bidirectional edge.
 * write_weight != 0  -> also store 'weight' in receiver's weights[rslot] (Option A, for FFNN)
 * write_weight == 0  -> pure topology (for GNG / self-organizing)
 */
static int add_bidirectional_edge(sonn_t *s, int from, int to, float weight, int write_weight)
{
    if (!s || !s->pool) return -1;
    if (from < 0 || to < 0 || from >= s->pool->max_neurons || to >= s->pool->max_neurons || from == to) return -1;

    neuron_t *nf = nnpool_get_neuron(s->pool, from);
    neuron_t *nt = nnpool_get_neuron(s->pool, to);
    if (!nf || !nt || !nf->active || !nt->active) return -1;

    int md = s->pool->max_degree;
    int fwd_slot = nnpool_find_edge_slot(s->pool, from, to);

    if (fwd_slot >= 0) {
        /* Connection already exists — refresh metadata */
        int eidx = from * md + fwd_slot;
        s->pool->edges[eidx].weight = 0.0f;
        s->pool->edges[eidx].age = 0;

        if (write_weight) {
            int rslot = nnpool_find_edge_slot(s->pool, to, from);
            if (rslot >= 0) {
                float *to_w = nnpool_get_params(s->pool, to);
                if (to_w) to_w[rslot + 1] = weight;
            }
        }
        return eidx;
    }

    if (s->pool->degrees[from] >= md) return -1;

    edge_t *row = nnpool_edge_row(s->pool, from);
    int slot = find_free_edge_slot(row, md);
    if (slot < 0) return -1;

    int eidx = from * md + slot;
    s->pool->degrees[from]++;

    edge_t *e = &s->pool->edges[eidx];
    e->from = from;
    e->to = to;
    e->weight = 0.0f;
    e->age = 0;
    e->active = 1;

    /* Always try to create reverse for easy neighbor walking */
    int rslot = nnpool_find_edge_slot(s->pool, to, from);
    if (rslot < 0 && s->pool->degrees[to] < md) {
        edge_t *rrow = nnpool_edge_row(s->pool, to);
        rslot = find_free_edge_slot(rrow, md);
        if (rslot >= 0) {
            s->pool->degrees[to]++;
            int re = to * md + rslot;
            edge_t *re_e = &s->pool->edges[re];
            re_e->from = to;
            re_e->to = from;
            re_e->weight = 0.0f;
            re_e->age = 0;
            re_e->active = 1;

            if (write_weight) {
                float *to_w = nnpool_get_params(s->pool, to);
                if (to_w) to_w[rslot + 1] = weight;
            }
        }
    } else if (rslot >= 0 && write_weight) {
        float *to_w = nnpool_get_params(s->pool, to);
        if (to_w) to_w[rslot + 1] = weight;
    }

    return eidx;
}

static void remove_bidirectional_edge(sonn_t *s, int from, int to, int clear_weight)
{
    if (!s || !s->pool) return;
    int md = s->pool->max_degree;

    int slot = nnpool_find_edge_slot(s->pool, from, to);
    if (slot >= 0) {
        edge_t *row = nnpool_edge_row(s->pool, from);
        row[slot].active = 0;
        row[slot].to = -1;
        s->pool->degrees[from]--;
        s->pool->edges[from * md + slot].active = 0;
    }

    int rslot = nnpool_find_edge_slot(s->pool, to, from);
    if (rslot >= 0) {
        edge_t *rrow = nnpool_edge_row(s->pool, to);
        rrow[rslot].active = 0;
        rrow[rslot].to = -1;
        s->pool->degrees[to]--;
        s->pool->edges[to * md + rslot].active = 0;

        if (clear_weight) {
            clear_weight_slot(s->pool, to, rslot);
        }
    }
}

/* Activate a freshly-acquired slot, initialize its edge row/degree, and stamp
 * the activation type. Used by sonn_create() and sonn_add_neuron().
 */
static int acquire_neuron_slot(sonn_t *s, activaton_t type)
{
    if (!s || !s->pool) return -1;

    int slot = nnpool_acquire_slot(s->pool);
    if (slot < 0) return -1;

    neuron_t *n = nnpool_get_neuron(s->pool, slot);
    float *params = nnpool_get_params(s->pool, slot);

    edge_t *erow = nnpool_edge_row(s->pool, slot);
    if (erow) {
        for (int i = 0; i < s->pool->max_degree; i++) {
            erow[i].active = 0;
            erow[i].to = -1;
        }
    }
    s->pool->degrees[slot] = 0;

    if (n) {
        neuron_activate(n, type, params, s->pool->input_dim);
    }

    s->current_neurons++;
    return slot;
}

sonn_t* sonn_create(int input_dim, int output_dim, int max_neurons, int max_degree)
{
    if (input_dim <= 0)  return NULL;
    if (output_dim <= 0) return NULL;
    if (max_neurons <= 0) return NULL;
    if (input_dim + output_dim >= max_neurons) return NULL;  /* need free room for growth */
    if (max_degree <= 0) max_degree = SONN_DEFAULT_MAX_DEGREE;

    sonn_t *s = (sonn_t*)calloc(1, sizeof(sonn_t));
    if (!s) return NULL;

    s->pool = nnpool_create(max_neurons, input_dim, max_degree);
    if (!s->pool) {
        free(s);
        return NULL;
    }

    s->max_neurons = max_neurons;
    s->max_degree = max_degree;
    s->input_dim = input_dim;
    s->output_dim = output_dim;

    /* Pre-create the input anchor neurons (slots 0 .. input_dim-1). */
    s->in_start = 0;
    s->in_count = 0;
    for (int i = 0; i < input_dim; i++) {
        int slot = acquire_neuron_slot(s, GELU);
        if (slot < 0) {
            sonn_destroy(s);
            return NULL;
        }
        /* Expectation from nnpool: first input_dim acquisitions come back as
         * slots 0..input_dim-1 in order. If a future allocator changes order,
         * this assertion-style guard catches it. */
        s->in_count++;
    }

    /* Pre-create the output anchor neurons (slots in_count .. in_count+output_dim-1). */
    s->out_start = s->in_count;
    s->out_count = 0;
    for (int i = 0; i < output_dim; i++) {
        int slot = acquire_neuron_slot(s, GELU);
        if (slot < 0) {
            sonn_destroy(s);
            return NULL;
        }
        s->out_count++;
    }

    return s;
}

void sonn_destroy(sonn_t *s)
{
    if (!s) return;
    if (s->pool) {
        nnpool_destroy(s->pool);
    }
    free(s);
}

int sonn_add_neuron(sonn_t *s, activaton_t type)
{
    return acquire_neuron_slot(s, type);
}

void sonn_remove_neuron(sonn_t *s, int id)
{
    if (!s || !s->pool) return;
    if (id < 0 || id >= s->pool->max_neurons) return;
    /* Refuse to remove the fixed input/output anchors — those live for the
     * lifetime of the SONN. */
    if (!sonn_is_interior(s, id)) return;

    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->active) return;

    neuron_deactivate(n);

    /* Remove all incident edges (semantic cleanup). */
    int md = s->pool->max_degree;
    edge_t *erow = nnpool_edge_row(s->pool, id);
    if (erow) {
        for (int i = 0; i < md; i++) {
            if (erow[i].active) {
                int to = erow[i].to;
                sonn_remove_edge(s, id, to);
            }
        }
    }

    /* Clean incoming edges from other neurons (reverse direction). */
    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (i == id) continue;
        neuron_t *other = nnpool_get_neuron(s->pool, i);
        if (!other || !other->active) continue;
        edge_t *orow = nnpool_edge_row(s->pool, i);
        if (!orow) continue;
        for (int j = 0; j < md; j++) {
            if (orow[j].active && orow[j].to == id) {
                sonn_remove_edge(s, i, id);
            }
        }
    }

    nnpool_release_slot(s->pool, id);
    s->current_neurons--;
}

int sonn_add_edge(sonn_t *s, int from, int to, float weight)
{
    return add_bidirectional_edge(s, from, to, weight, 1);
}

int sonn_add_edge_topology(sonn_t *s, int from, int to)
{
    return add_bidirectional_edge(s, from, to, 0.0f, 0);
}

void sonn_remove_edge(sonn_t *s, int from, int to)
{
    remove_bidirectional_edge(s, from, to, 1);
}

int sonn_is_interior(sonn_t *s, int id)
{
    if (!s || id < 0) return 0;
    if (id >= s->in_start && id < s->in_start + s->in_count) return 0;
    if (id >= s->out_start && id < s->out_start + s->out_count) return 0;
    return 1;
}

int sonn_get_neighbors(sonn_t *s, int id, int *out, int max_out)
{
    if (!s || !s->pool || id < 0 || !out || max_out <= 0) return 0;
    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->active) return 0;

    int md = s->pool->max_degree;
    edge_t *erow = nnpool_edge_row(s->pool, id);
    if (!erow) return 0;

    int count = 0;
    for (int i = 0; i < md && count < max_out; i++) {
        if (erow[i].active) out[count++] = erow[i].to;
    }
    return count;
}

int sonn_get_neuronid(sonn_t *s, int slot)
{
    if (!s || !s->pool) return 0;
    neuron_t *n = nnpool_get_neuron(s->pool, slot);
    return n ? n->id : 0;
}

int sonn_get_input_range(sonn_t *s, int *start, int *count)
{
    if (!s) return -1;
    if (start) *start = s->in_start;
    if (count) *count = s->in_count;
    return 0;
}

int sonn_get_output_range(sonn_t *s, int *start, int *count)
{
    if (!s) return -1;
    if (start) *start = s->out_start;
    if (count) *count = s->out_count;
    return 0;
}

int sonn_get_output(sonn_t *s, float *out)
{
    if (!s || !s->pool || !out) return -1;
    for (int i = 0; i < s->out_count; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, s->out_start + i);
        out[i] = n ? n->activation : 0.0f;
    }
    return 0;
}