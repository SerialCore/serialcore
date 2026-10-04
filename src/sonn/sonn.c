/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/sonn.h>
#include <serialcore/iobin.h>
#include <serialcore/iojson.h>
#include <serialcore/types.h>

#include <stdio.h>
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

    /* Already linked. Symmetric storage keeps the matching half. Age is not ours. */
    int slot_a = nnpool_find_edge_slot(s->pool, a, b);
    if (slot_a >= 0) {
        return a * md + slot_a;
    }

    /* Need room in both rows; refuse if either endpoint is already full. */
    if (s->pool->degrees[a] >= md) return -1;
    if (s->pool->degrees[b] >= md) return -1;

    /* First free slot in each row. */
    edge_t *row_a = nnpool_edge_row(s->pool, a);
    int slot_a_free = -1;
    for (int k = 0; k < md; k++) {
        if (!row_a[k].active) { slot_a_free = k; break; }
    }
    if (slot_a_free < 0) return -1;

    edge_t *row_b = nnpool_edge_row(s->pool, b);
    int slot_b_free = -1;
    for (int k = 0; k < md; k++) {
        if (!row_b[k].active) { slot_b_free = k; break; }
    }
    if (slot_b_free < 0) return -1;

    /* Add a new connectivity in free slot */
    int eidx_a = a * md + slot_a_free;
    int eidx_b = b * md + slot_b_free;

    edge_t *ea = &s->pool->edges[eidx_a];
    ea->from = a;
    ea->to = b;
    ea->active = 1;
    s->pool->degrees[a]++;

    edge_t *eb = &s->pool->edges[eidx_b];
    eb->from = b;
    eb->to = a;
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

int sonn_is_interior(const sonn_t *s, int id)
{
    if (!s || id < 0) return 0;
    if (id >= s->in_start && id < s->in_start + s->in_count) return 0;
    if (id >= s->out_start && id < s->out_start + s->out_count) return 0;
    return 1;
}

int sonn_get_neighbors(const sonn_t *s, int id, int *out, int max_out)
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

int sonn_get_input_range(const sonn_t *s, int *start, int *count)
{
    if (!s) return -1;
    if (start) *start = s->in_start;
    if (count) *count = s->in_count;
    return 0;
}

int sonn_get_output_range(const sonn_t *s, int *start, int *count)
{
    if (!s) return -1;
    if (start) *start = s->out_start;
    if (count) *count = s->out_count;
    return 0;
}

int sonn_foreach_neighbor(const sonn_t *s, int id, sonn_neighbor_cb cb, void *userdata)
{
    if (!s || !s->pool || !cb || id < 0) return 0;
    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->active) return 0;

    edge_t *erow = nnpool_edge_row(s->pool, id);
    if (!erow) return 0;

    int seen = 0;
    int md = s->pool->max_degree;
    for (int i = 0; i < md; i++) {
        if (!erow[i].active) continue;
        seen++;
        if (cb(erow[i].to, userdata)) return seen;
    }
    return seen;
}

/* Claim/activate a neuron slot described by meta JSON (id + type only). */
static int sonn_claim_neuron(sonn_t *s, const cJSON *jn, int expect_existing)
{
    int id = 0;
    const char *type_s = NULL;
    activaton_t type;
    neuron_t *n;
    float *params;

    if (!s || !jn) return -1;
    if (iojson_get_int(jn, "id", &id) != 0) return -1;
    if (iojson_get_string(jn, "type", &type_s) != 0) return -1;

    type = activaton_type((char *)type_s);
    n = nnpool_get_neuron(s->pool, id);

    if (expect_existing) {
        if (!n || !n->active) return -1;
        n->type = type;
        return 0;
    }

    if (n && n->active) return -1;
    if (nnpool_claim_slot(s->pool, id) != id) return -1;

    n = nnpool_get_neuron(s->pool, id);
    params = nnpool_get_params(s->pool, id);
    {
        edge_t *erow = nnpool_edge_row(s->pool, id);
        if (erow) {
            for (int i = 0; i < s->pool->max_degree; i++) {
                erow[i].active = 0;
                erow[i].to = -1;
            }
        }
        s->pool->degrees[id] = 0;
    }
    if (n) neuron_activate(n, type, params, s->pool->input_dim);
    s->current_neurons++;
    return 0;
}

cJSON *sonn_to_json(const sonn_t *s)
{
    cJSON *root;
    cJSON *neurons;
    cJSON *edges;
    activaton_t default_type = GELU;
    int md;

    if (!s || !s->pool || !s->pool->params) return NULL;

    root = cJSON_CreateObject();
    if (!root) return NULL;

    {
        neuron_t *n0 = nnpool_get_neuron(s->pool, s->in_start);
        if (n0 && n0->active) default_type = n0->type;
    }

    cJSON_AddStringToObject(root, "kind", nn_kind_str[NN_KIND_SONN]);
    cJSON_AddNumberToObject(root, "input_dim", s->input_dim);
    cJSON_AddNumberToObject(root, "output_dim", s->output_dim);
    cJSON_AddNumberToObject(root, "max_neurons", s->max_neurons);
    cJSON_AddNumberToObject(root, "max_degree", s->max_degree);
    cJSON_AddNumberToObject(root, "in_start", s->in_start);
    cJSON_AddNumberToObject(root, "in_count", s->in_count);
    cJSON_AddNumberToObject(root, "out_start", s->out_start);
    cJSON_AddNumberToObject(root, "out_count", s->out_count);
    cJSON_AddNumberToObject(root, "current_neurons", s->current_neurons);
    cJSON_AddStringToObject(root, "activation", activaton_name(default_type));

    neurons = cJSON_CreateArray();
    edges = cJSON_CreateArray();
    if (!neurons || !edges) {
        cJSON_Delete(neurons);
        cJSON_Delete(edges);
        cJSON_Delete(root);
        return NULL;
    }
    cJSON_AddItemToObject(root, "neurons", neurons);
    cJSON_AddItemToObject(root, "edges", edges);

    md = s->pool->max_degree;
    for (int id = 0; id < s->pool->max_neurons; id++) {
        neuron_t *n = nnpool_get_neuron(s->pool, id);
        cJSON *jn;
        edge_t *row;

        if (!n || !n->active) continue;

        jn = cJSON_CreateObject();
        if (!jn) {
            cJSON_Delete(root);
            return NULL;
        }
        cJSON_AddNumberToObject(jn, "id", id);
        cJSON_AddStringToObject(jn, "type", activaton_name(n->type));
        cJSON_AddItemToArray(neurons, jn);

        /* Emit each undirected edge once (to > id). */
        row = nnpool_edge_row(s->pool, id);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            cJSON *je;
            if (!row[j].active || row[j].to <= id) continue;
            je = cJSON_CreateObject();
            if (!je) {
                cJSON_Delete(root);
                return NULL;
            }
            cJSON_AddNumberToObject(je, "from", id);
            cJSON_AddNumberToObject(je, "to", row[j].to);
            cJSON_AddItemToArray(edges, je);
        }
    }

    return root;
}

sonn_t *sonn_from_json(const cJSON *root)
{
    const char *act_s = NULL;
    int input_dim = 0, output_dim = 0, max_neurons = 0, max_degree = 0;
    int in_start = 0, in_count = 0, out_start = 0, out_count = 0;
    activaton_t default_type = GELU;
    cJSON *neurons;
    cJSON *edges;
    sonn_t *s = NULL;
    int ok = 1;
    int n_neurons;

    if (!root || !cJSON_IsObject(root)) return NULL;

    if (iojson_get_int(root, "input_dim", &input_dim) != 0 ||
        iojson_get_int(root, "output_dim", &output_dim) != 0 ||
        iojson_get_int(root, "max_neurons", &max_neurons) != 0 ||
        iojson_get_int(root, "max_degree", &max_degree) != 0) {
        return NULL;
    }

    iojson_get_int(root, "in_start", &in_start);
    iojson_get_int(root, "in_count", &in_count);
    iojson_get_int(root, "out_start", &out_start);
    iojson_get_int(root, "out_count", &out_count);

    if (iojson_get_string(root, "activation", &act_s) == 0) {
        default_type = activaton_type((char *)act_s);
    }

    neurons = iojson_get_array(root, "neurons");
    edges = iojson_get_array(root, "edges");
    if (!neurons) return NULL;

    s = sonn_create(input_dim, output_dim, max_neurons, max_degree, default_type);
    if (!s) return NULL;

    if (in_count > 0 && (s->in_start != in_start || s->in_count != in_count)) ok = 0;
    if (ok && out_count > 0 && (s->out_start != out_start || s->out_count != out_count)) ok = 0;

    n_neurons = cJSON_GetArraySize(neurons);
    for (int i = 0; ok && i < n_neurons; i++) {
        cJSON *jn = cJSON_GetArrayItem(neurons, i);
        int id = 0;
        int is_anchor;

        if (!cJSON_IsObject(jn) || iojson_get_int(jn, "id", &id) != 0) {
            ok = 0;
            break;
        }

        is_anchor = (id >= s->in_start && id < s->in_start + s->in_count) ||
                    (id >= s->out_start && id < s->out_start + s->out_count);

        if (sonn_claim_neuron(s, jn, is_anchor) != 0) ok = 0;
    }

    if (ok && edges) {
        int n_edges = cJSON_GetArraySize(edges);
        for (int i = 0; ok && i < n_edges; i++) {
            cJSON *je = cJSON_GetArrayItem(edges, i);
            int from = 0, to = 0;

            if (!cJSON_IsObject(je) ||
                iojson_get_int(je, "from", &from) != 0 ||
                iojson_get_int(je, "to", &to) != 0) {
                ok = 0;
                break;
            }

            /* Older files may still carry "age"; the graph ignores it. */
            if (sonn_add_edge(s, from, to) < 0) {
                ok = 0;
                break;
            }
        }
    }

    if (!ok) {
        sonn_destroy(s);
        return NULL;
    }
    return s;
}

int sonn_save(const sonn_t *s, const char *json_path, const char *bin_path)
{
    cJSON *root;
    int nparams;
    int rc;

    if (!s || !s->pool || !s->pool->params || !json_path || !bin_path) return -1;

    root = sonn_to_json(s);
    if (!root) return -1;
    rc = iojson_write_cjson(json_path, root);
    cJSON_Delete(root);
    if (rc != 0) return -1;

    nparams = s->pool->max_neurons * (s->pool->input_dim + 1);
    return iobin_write_floats(bin_path, s->pool->params, nparams);
}

sonn_t *sonn_load(const char *json_path, const char *bin_path)
{
    cJSON *root;
    sonn_t *s;
    int nparams;

    if (!json_path || !bin_path) return NULL;

    root = iojson_parse_file(json_path);
    if (!root) return NULL;

    s = sonn_from_json(root);
    cJSON_Delete(root);
    if (!s) return NULL;

    nparams = s->pool->max_neurons * (s->pool->input_dim + 1);
    if (iobin_read_floats(bin_path, s->pool->params, nparams) != 0) {
        sonn_destroy(s);
        return NULL;
    }
    sonn_rebind_params(s);
    return s;
}

void sonn_rebind_params(sonn_t *s)
{
    if (!s || !s->pool) return;
    for (int id = 0; id < s->pool->max_neurons; id++) {
        neuron_t *n = nnpool_get_neuron(s->pool, id);
        float *params;
        if (!n || !n->active) continue;
        params = nnpool_get_params(s->pool, id);
        if (!params) continue;
        n->bias = params[0];
        n->weights = params + 1;
    }
}