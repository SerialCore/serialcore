/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/sonn.h>

#include <stdlib.h>
#include <string.h>

sonn_t* sonn_create(int input_dim, int output_dim, int max_neurons, int max_degree, activaton_t type)
{
    if (input_dim <= 0)  return NULL;
    if (output_dim <= 0) return NULL;
    if (max_neurons <= 0) return NULL;
    if (input_dim + output_dim >= max_neurons) return NULL;
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
        int slot = sonn_add_neuron(s, type);
        if (slot < 0) {
            sonn_destroy(s);
            return NULL;
        }
        s->in_count++;
    }

    /* Pre-create the output anchor neurons (slots in_count .. in_count+output_dim-1). */
    s->out_start = s->in_count;
    s->out_count = 0;
    for (int i = 0; i < output_dim; i++) {
        int slot = sonn_add_neuron(s, type);
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

void sonn_remove_neuron(sonn_t *s, int id)
{
    if (!s || !s->pool) return;
    if (id < 0 || id >= s->pool->max_neurons) return;
    /* Refuse to remove the fixed input/output anchors — those live for the lifetime of the SONN. */
    if (!sonn_is_interior(s, id)) return;

    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->active) return;

    neuron_deactivate(n);

    int md = s->pool->max_degree;
    edge_t *erow = nnpool_edge_row(s->pool, id);
    if (erow) {
        for (int i = 0; i < md; i++) {
            if (erow[i].active) {
                int neighbor = erow[i].to;
                sonn_remove_edge(s, id, neighbor);
            }
        }
    }

    nnpool_release_slot(s->pool, id);
    s->current_neurons--;
}

int sonn_add_edge(sonn_t *s, int a, int b)
{
    if (!s || !s->pool) return -1;
    if (a < 0 || b < 0 || a >= s->pool->max_neurons || b >= s->pool->max_neurons || a == b) return -1;

    neuron_t *na = nnpool_get_neuron(s->pool, a);
    neuron_t *nb = nnpool_get_neuron(s->pool, b);
    if (!na || !nb || !na->active || !nb->active) return -1;

    int md = s->pool->max_degree;

    /* If the edge already exists (in a's row), refresh both halves' age and bail.
     * Symmetric storage means b's matching entry exists too. */
    int slot_a = nnpool_find_edge_slot(s->pool, a, b);
    if (slot_a >= 0) {
        int eidx_a = a * md + slot_a;
        s->pool->edges[eidx_a].age = 0;
        int slot_b = nnpool_find_edge_slot(s->pool, b, a);
        if (slot_b >= 0) {
            int eidx_b = b * md + slot_b;
            s->pool->edges[eidx_b].age = 0;
        }
        return eidx_a;
    }

    /* Need room in both rows; refuse if either endpoint is already full. */
    if (s->pool->degrees[a] >= md) return -1;
    if (s->pool->degrees[b] >= md) return -1;

    /* Search for the free slot in edge. */
    edge_t *row_a = nnpool_edge_row(s->pool, a);
    int slot_a_free = -1;
    for (int s = 0; s < md; s++) {
        if (!row_a[s].active) slot_a_free = s;
    }
    if (slot_a_free < 0) return -1;

    edge_t *row_b = nnpool_edge_row(s->pool, b);
    int slot_b_free = -1;
    for (int s = 0; s < md; s++) {
        if (!row_b[s].active) slot_b_free = s;
    }
    if (slot_b_free < 0) return -1;

    /* Add a new connectivity in free slot */
    int eidx_a = a * md + slot_a_free;
    int eidx_b = b * md + slot_b_free;

    edge_t *ea = &s->pool->edges[eidx_a];
    ea->from = a;
    ea->to = b;
    ea->age = 0;
    ea->active = 1;
    s->pool->degrees[a]++;

    edge_t *eb = &s->pool->edges[eidx_b];
    eb->from = b;
    eb->to = a;
    eb->age = 0;
    eb->active = 1;
    s->pool->degrees[b]++;

    return eidx_a;
}

void sonn_remove_edge(sonn_t *s, int a, int b)
{
    if (!s || !s->pool) return;

    int slot_a = nnpool_find_edge_slot(s->pool, a, b);
    if (slot_a >= 0) {
        edge_t *row_a = nnpool_edge_row(s->pool, a);
        row_a[slot_a].active = 0;
        row_a[slot_a].to = -1;
        s->pool->degrees[a]--;
    }

    int slot_b = nnpool_find_edge_slot(s->pool, b, a);
    if (slot_b >= 0) {
        edge_t *row_b = nnpool_edge_row(s->pool, b);
        row_b[slot_b].active = 0;
        row_b[slot_b].to = -1;
        s->pool->degrees[b]--;
    }
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