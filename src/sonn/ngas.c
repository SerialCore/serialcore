/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/ngas.h>
#include <serialcore/iobin.h>
#include <serialcore/iojson.h>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static float ngas_sqdist(const float *prototype, const float *x, int dim, void *userdata)
{
    float dist = 0.0f;
    (void)userdata;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - prototype[i];
        dist += d * d;
    }
    return dist;
}

static float sched_decay(float start, float end, int step, int n_steps)
{
    if (step < 0) step = 0;
    if (!(start > 0.0f) || !(end > 0.0f)) return end > 0.0f ? end : start;
    if (n_steps <= 1) return (step == 0) ? start : end;
    if (step >= n_steps - 1) return end;
    return start * powf(end / start, (float)step / (float)(n_steps - 1));
}

static void ngas_apply_schedule(ngas_t *m, int n_steps, float eps_start, float eps_end,
                                float lambda_start, float lambda_end)
{
    m->n_steps = n_steps > 0 ? n_steps : NGAS_DEFAULT_N_STEPS;
    m->eps_start = eps_start > 0.0f ? eps_start : NGAS_DEFAULT_EPS_START;
    m->eps_end = eps_end > 0.0f ? eps_end : NGAS_DEFAULT_EPS_END;
    m->lambda_start = lambda_start > 0.0f ? lambda_start : (float)m->n_units;
    m->lambda_end = lambda_end > 0.0f ? lambda_end : NGAS_DEFAULT_LAMBDA_END;
}

static ngas_t *ngas_alloc(sonn_t *net, int n_units)
{
    ngas_t *m;

    if (!net || !net->pool || n_units <= 0) return NULL;

    m = (ngas_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->net = net;
    m->n_units = n_units;
    m->distance = ngas_sqdist;
    m->unit_ids = (int *)malloc((size_t)n_units * sizeof(int));
    if (!m->unit_ids) {
        ngas_destroy(m);
        return NULL;
    }
    for (int i = 0; i < n_units; i++) m->unit_ids[i] = -1;
    return m;
}

static int ngas_bind_units(ngas_t *m, const int *ids)
{
    sonn_t *s = m->net;
    for (int k = 0; k < m->n_units; k++) {
        int id = ids[k];
        neuron_t *n;
        if (id < 0 || id >= s->max_neurons) return -1;
        n = nnpool_get_neuron(s->pool, id);
        if (!n || !n->active || !sonn_is_interior(s, id)) return -1;
        for (int j = 0; j < k; j++) {
            if (m->unit_ids[j] == id) return -1;
        }
        m->unit_ids[k] = id;
    }
    return 0;
}

static void ngas_rollback_units(ngas_t *m, int added)
{
    for (int j = 0; j < added; j++) {
        if (m->unit_ids[j] >= 0) sonn_remove_neuron(m->net, m->unit_ids[j]);
    }
}

ngas_t *ngas_create(sonn_t *net, int n_units, int n_steps,
                    float eps_start, float eps_end, float lambda_start, float lambda_end)
{
    ngas_t *m = ngas_alloc(net, n_units);
    neuron_t *anchor;
    activaton_t type;

    if (!m) return NULL;
    ngas_apply_schedule(m, n_steps, eps_start, eps_end, lambda_start, lambda_end);

    if (net->pool->free_count < m->n_units) {
        ngas_destroy(m);
        return NULL;
    }

    anchor = nnpool_get_neuron(net->pool, net->in_start);
    type = (anchor && anchor->active) ? anchor->type : GELU;

    for (int k = 0; k < m->n_units; k++) {
        int id = sonn_add_neuron(net, type);
        if (id < 0) {
            ngas_rollback_units(m, k);
            ngas_destroy(m);
            return NULL;
        }
        m->unit_ids[k] = id;
    }
    return m;
}

void ngas_destroy(ngas_t *m)
{
    if (!m) return;
    free(m->unit_ids);
    free(m);
}

void ngas_configure(ngas_t *m, int n_steps, float eps_start, float eps_end,
                    float lambda_start, float lambda_end)
{
    if (!m) return;
    ngas_apply_schedule(m, n_steps, eps_start, eps_end, lambda_start, lambda_end);
}

void ngas_set_distance(ngas_t *m, ngas_distance_fn fn, void *userdata)
{
    if (!m) return;
    m->distance = fn ? fn : ngas_sqdist;
    m->distance_userdata = fn ? userdata : NULL;
}

float ngas_epsilon(const ngas_t *m)
{
    if (!m) return 0.0f;
    return sched_decay(m->eps_start, m->eps_end, m->observe_count, m->n_steps);
}

float ngas_lambda(const ngas_t *m)
{
    if (!m) return 0.0f;
    return sched_decay(m->lambda_start, m->lambda_end, m->observe_count, m->n_steps);
}

int ngas_unit_at(const ngas_t *m, int index)
{
    if (!m || index < 0 || index >= m->n_units) return -1;
    return m->unit_ids[index];
}

float ngas_prototype_distance(ngas_t *m, int neuron_id, const float *input)
{
    neuron_t *n;
    ngas_distance_fn fn;

    if (!m || !m->net || !m->net->pool || !input || neuron_id < 0) return -1.0f;
    if (neuron_id >= m->net->max_neurons) return -1.0f;

    n = nnpool_get_neuron(m->net->pool, neuron_id);
    if (!n || !n->active || !n->weights) return -1.0f;

    fn = m->distance ? m->distance : ngas_sqdist;
    return fn(n->weights, input, n->input_dim, m->distance_userdata);
}

int ngas_find_bmu(ngas_t *m, const float *input)
{
    int best = -1;
    float best_dist = 1e30f;

    if (!m || !input) return -1;
    for (int k = 0; k < m->n_units; k++) {
        float dist = ngas_prototype_distance(m, m->unit_ids[k], input);
        if (dist < 0.0f) continue;
        if (dist < best_dist) {
            best_dist = dist;
            best = m->unit_ids[k];
        }
    }
    return best;
}

static void move_toward(neuron_t *n, const float *x, float coeff)
{
    if (!n || !n->weights || coeff == 0.0f) return;
    for (int i = 0; i < n->input_dim; i++) {
        n->weights[i] += coeff * (x[i] - n->weights[i]);
    }
}

int ngas_observe(ngas_t *m, const float *x)
{
    float eps;
    float lambda;
    float *dist;
    sonn_t *s;

    if (!m || !m->net || !m->net->pool || !x || m->n_units <= 0) return -1;

    dist = (float *)malloc((size_t)m->n_units * sizeof(float));
    if (!dist) return -1;

    s = m->net;
    for (int i = 0; i < m->n_units; i++) {
        dist[i] = ngas_prototype_distance(m, m->unit_ids[i], x);
        if (dist[i] < 0.0f) {
            free(dist);
            return -1;
        }
    }

    eps = ngas_epsilon(m);
    lambda = ngas_lambda(m);
    if (lambda < 1e-12f) lambda = 1e-12f;

    for (int i = 0; i < m->n_units; i++) {
        int rank = 0;
        neuron_t *n;
        float h;

        for (int j = 0; j < m->n_units; j++) {
            if (dist[j] < dist[i]) rank++;
        }
        n = nnpool_get_neuron(s->pool, m->unit_ids[i]);
        h = expf(-(float)rank / lambda);
        move_toward(n, x, eps * h);
    }

    free(dist);
    m->observe_count++;
    return 0;
}

static cJSON *ngas_state_json(const ngas_t *m)
{
    cJSON *nj = cJSON_CreateObject();
    cJSON *units = cJSON_CreateArray();
    if (!nj || !units) {
        cJSON_Delete(nj);
        cJSON_Delete(units);
        return NULL;
    }

    cJSON_AddNumberToObject(nj, "n_units", m->n_units);
    cJSON_AddNumberToObject(nj, "n_steps", m->n_steps);
    cJSON_AddNumberToObject(nj, "observe_count", m->observe_count);
    cJSON_AddNumberToObject(nj, "eps_start", m->eps_start);
    cJSON_AddNumberToObject(nj, "eps_end", m->eps_end);
    cJSON_AddNumberToObject(nj, "lambda_start", m->lambda_start);
    cJSON_AddNumberToObject(nj, "lambda_end", m->lambda_end);
    cJSON_AddItemToObject(nj, "units", units);

    for (int k = 0; k < m->n_units; k++) {
        cJSON *id = cJSON_CreateNumber(m->unit_ids[k]);
        if (!id) {
            cJSON_Delete(nj);
            return NULL;
        }
        cJSON_AddItemToArray(units, id);
    }
    return nj;
}

int ngas_save(const ngas_t *m, const char *json_path, const char *bin_path)
{
    cJSON *root;
    cJSON *state;
    int nparams;
    int rc;

    if (!m || !m->net || !m->net->pool || !m->net->pool->params || !json_path || !bin_path) {
        return -1;
    }

    root = sonn_to_json(m->net);
    state = ngas_state_json(m);
    if (!root || !state) {
        cJSON_Delete(root);
        cJSON_Delete(state);
        return -1;
    }
    cJSON_AddItemToObject(root, "ngas", state);

    rc = iojson_write_cjson(json_path, root);
    cJSON_Delete(root);
    if (rc != 0) return -1;

    nparams = m->net->pool->max_neurons * (m->net->pool->input_dim + 1);
    return iobin_write_floats(bin_path, m->net->pool->params, nparams);
}

ngas_t *ngas_load(const char *json_path, const char *bin_path, sonn_t **net_out)
{
    cJSON *root;
    cJSON *nj;
    cJSON *units;
    sonn_t *s;
    ngas_t *m;
    int *ids = NULL;
    int nparams;
    int n_units = 0, n_steps = NGAS_DEFAULT_N_STEPS, observe_count = 0;
    float eps_start = NGAS_DEFAULT_EPS_START, eps_end = NGAS_DEFAULT_EPS_END;
    float lambda_start = 0.0f, lambda_end = NGAS_DEFAULT_LAMBDA_END;

    if (!json_path || !bin_path || !net_out) return NULL;
    *net_out = NULL;

    root = iojson_parse_file(json_path);
    if (!root || !cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return NULL;
    }

    s = sonn_from_json(root);
    if (!s) {
        cJSON_Delete(root);
        return NULL;
    }

    nparams = s->pool->max_neurons * (s->pool->input_dim + 1);
    if (iobin_read_floats(bin_path, s->pool->params, nparams) != 0) {
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }
    sonn_rebind_params(s);

    nj = iojson_get_object(root, "ngas");
    units = nj ? iojson_get_array(nj, "units") : NULL;
    if (!nj || !units || iojson_get_int(nj, "n_units", &n_units) != 0) {
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    iojson_get_int(nj, "n_steps", &n_steps);
    iojson_get_int(nj, "observe_count", &observe_count);
    iojson_get_float(nj, "eps_start", &eps_start);
    iojson_get_float(nj, "eps_end", &eps_end);
    iojson_get_float(nj, "lambda_start", &lambda_start);
    iojson_get_float(nj, "lambda_end", &lambda_end);

    m = ngas_alloc(s, n_units);
    if (!m || cJSON_GetArraySize(units) != m->n_units) {
        ngas_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    m->n_steps = n_steps;
    m->observe_count = observe_count;
    m->eps_start = eps_start;
    m->eps_end = eps_end;
    m->lambda_start = lambda_start;
    m->lambda_end = lambda_end;

    ids = (int *)malloc((size_t)m->n_units * sizeof(int));
    if (!ids) {
        ngas_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }
    for (int k = 0; k < m->n_units; k++) {
        cJSON *item = cJSON_GetArrayItem(units, k);
        if (!cJSON_IsNumber(item)) {
            free(ids);
            ngas_destroy(m);
            sonn_destroy(s);
            cJSON_Delete(root);
            return NULL;
        }
        ids[k] = (int)item->valuedouble;
    }
    if (ngas_bind_units(m, ids) != 0) {
        free(ids);
        ngas_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    free(ids);
    cJSON_Delete(root);
    *net_out = s;
    return m;
}
