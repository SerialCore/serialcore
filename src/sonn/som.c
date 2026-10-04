/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/som.h>
#include <serialcore/iobin.h>
#include <serialcore/iojson.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static float som_sqdist(const float *prototype, const float *x, int dim, void *userdata)
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

static float default_sigma_start(const som_t *m)
{
    int span = m->rows > m->cols ? m->rows : m->cols;
    if (span < 2) return 1.0f;
    return 0.5f * (float)span;
}

static void som_apply_schedule(som_t *m, int n_steps, float eps_start, float eps_end,
                               float sigma_start, float sigma_end)
{
    m->n_steps = n_steps > 0 ? n_steps : SOM_DEFAULT_N_STEPS;
    m->eps_start = eps_start > 0.0f ? eps_start : SOM_DEFAULT_EPS_START;
    m->eps_end = eps_end > 0.0f ? eps_end : SOM_DEFAULT_EPS_END;
    m->sigma_start = sigma_start > 0.0f ? sigma_start : default_sigma_start(m);
    m->sigma_end = sigma_end > 0.0f ? sigma_end : SOM_DEFAULT_SIGMA_END;
}

static som_t *som_alloc(sonn_t *net, int rows, int cols)
{
    if (!net || !net->pool || rows <= 0 || cols <= 0) return NULL;
    if (rows > INT_MAX / cols) return NULL;

    som_t *m = (som_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->net = net;
    m->rows = rows;
    m->cols = cols;
    m->n_units = rows * cols;
    m->distance = som_sqdist;

    m->unit_ids = (int *)calloc((size_t)m->n_units, sizeof(int));
    m->row_of = (int *)malloc((size_t)net->pool->max_neurons * sizeof(int));
    m->col_of = (int *)malloc((size_t)net->pool->max_neurons * sizeof(int));
    if (!m->unit_ids || !m->row_of || !m->col_of) {
        som_destroy(m);
        return NULL;
    }

    for (int i = 0; i < net->pool->max_neurons; i++) {
        m->row_of[i] = -1;
        m->col_of[i] = -1;
    }
    for (int k = 0; k < m->n_units; k++) m->unit_ids[k] = -1;
    return m;
}

static void som_place(som_t *m, int k, int id)
{
    m->unit_ids[k] = id;
    m->row_of[id] = k / m->cols;
    m->col_of[id] = k % m->cols;
}

static int som_bind_units(som_t *m, const int *ids)
{
    sonn_t *s = m->net;
    for (int k = 0; k < m->n_units; k++) {
        int id = ids[k];
        neuron_t *n;
        if (id < 0 || id >= s->max_neurons) return -1;
        n = nnpool_get_neuron(s->pool, id);
        if (!n || !n->active || !sonn_is_interior(s, id)) return -1;
        if (m->row_of[id] >= 0) return -1;
        som_place(m, k, id);
    }
    return 0;
}

static void som_rollback_units(som_t *m, int added)
{
    for (int j = 0; j < added; j++) {
        if (m->unit_ids[j] >= 0) sonn_remove_neuron(m->net, m->unit_ids[j]);
    }
}

som_t *som_create(sonn_t *net, int rows, int cols, int n_steps,
                  float eps_start, float eps_end, float sigma_start, float sigma_end)
{
    som_t *m = som_alloc(net, rows, cols);
    neuron_t *anchor;
    activaton_t type;

    if (!m) return NULL;
    som_apply_schedule(m, n_steps, eps_start, eps_end, sigma_start, sigma_end);

    if (net->pool->free_count < m->n_units) {
        som_destroy(m);
        return NULL;
    }

    anchor = nnpool_get_neuron(net->pool, net->in_start);
    type = (anchor && anchor->active) ? anchor->type : GELU;

    for (int k = 0; k < m->n_units; k++) {
        int id = sonn_add_neuron(net, type);
        if (id < 0) {
            som_rollback_units(m, k);
            som_destroy(m);
            return NULL;
        }
        som_place(m, k, id);
    }
    return m;
}

void som_destroy(som_t *m)
{
    if (!m) return;
    free(m->unit_ids);
    free(m->row_of);
    free(m->col_of);
    free(m);
}

void som_configure(som_t *m, int n_steps, float eps_start, float eps_end,
                   float sigma_start, float sigma_end)
{
    if (!m) return;
    som_apply_schedule(m, n_steps, eps_start, eps_end, sigma_start, sigma_end);
}

void som_set_distance(som_t *m, som_distance_fn fn, void *userdata)
{
    if (!m) return;
    m->distance = fn ? fn : som_sqdist;
    m->distance_userdata = fn ? userdata : NULL;
}

float som_epsilon(const som_t *m)
{
    if (!m) return 0.0f;
    return sched_decay(m->eps_start, m->eps_end, m->observe_count, m->n_steps);
}

float som_sigma(const som_t *m)
{
    if (!m) return 0.0f;
    return sched_decay(m->sigma_start, m->sigma_end, m->observe_count, m->n_steps);
}

int som_unit_at(const som_t *m, int row, int col)
{
    if (!m || row < 0 || col < 0 || row >= m->rows || col >= m->cols) return -1;
    return m->unit_ids[row * m->cols + col];
}

float som_prototype_distance(som_t *m, int neuron_id, const float *input)
{
    neuron_t *n;
    som_distance_fn fn;

    if (!m || !m->net || !m->net->pool || !input || neuron_id < 0) return -1.0f;
    if (neuron_id >= m->net->max_neurons || m->row_of[neuron_id] < 0) return -1.0f;

    n = nnpool_get_neuron(m->net->pool, neuron_id);
    if (!n || !n->active || !n->weights) return -1.0f;

    fn = m->distance ? m->distance : som_sqdist;
    return fn(n->weights, input, n->input_dim, m->distance_userdata);
}

int som_find_bmu(som_t *m, const float *input)
{
    int best = -1;
    float best_dist = 1e30f;

    if (!m || !input) return -1;
    for (int k = 0; k < m->n_units; k++) {
        float dist = som_prototype_distance(m, m->unit_ids[k], input);
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

int som_observe(som_t *m, const float *x)
{
    int bmu;
    float eps;
    float sigma;
    float denom;
    sonn_t *s;

    if (!m || !m->net || !m->net->pool || !x) return -1;
    bmu = som_find_bmu(m, x);
    if (bmu < 0) return -1;

    eps = som_epsilon(m);
    sigma = som_sigma(m);
    denom = 2.0f * sigma * sigma;
    s = m->net;

    for (int k = 0; k < m->n_units; k++) {
        int id = m->unit_ids[k];
        neuron_t *n = nnpool_get_neuron(s->pool, id);
        int dr, dc, d2;
        float h;

        if (!n || !n->active) continue;
        dr = m->row_of[id] - m->row_of[bmu];
        dc = m->col_of[id] - m->col_of[bmu];
        d2 = dr * dr + dc * dc;
        if (denom < 1e-12f) h = (id == bmu) ? 1.0f : 0.0f;
        else                 h = expf(-(float)d2 / denom);
        move_toward(n, x, eps * h);
    }

    m->observe_count++;
    return 0;
}

static cJSON *som_state_json(const som_t *m)
{
    cJSON *sj = cJSON_CreateObject();
    cJSON *units = cJSON_CreateArray();
    if (!sj || !units) {
        cJSON_Delete(sj);
        cJSON_Delete(units);
        return NULL;
    }

    cJSON_AddNumberToObject(sj, "rows", m->rows);
    cJSON_AddNumberToObject(sj, "cols", m->cols);
    cJSON_AddNumberToObject(sj, "n_steps", m->n_steps);
    cJSON_AddNumberToObject(sj, "observe_count", m->observe_count);
    cJSON_AddNumberToObject(sj, "eps_start", m->eps_start);
    cJSON_AddNumberToObject(sj, "eps_end", m->eps_end);
    cJSON_AddNumberToObject(sj, "sigma_start", m->sigma_start);
    cJSON_AddNumberToObject(sj, "sigma_end", m->sigma_end);
    cJSON_AddItemToObject(sj, "units", units);

    for (int k = 0; k < m->n_units; k++) {
        cJSON *id = cJSON_CreateNumber(m->unit_ids[k]);
        if (!id) {
            cJSON_Delete(sj);
            return NULL;
        }
        cJSON_AddItemToArray(units, id);
    }
    return sj;
}

int som_save(const som_t *m, const char *json_path, const char *bin_path)
{
    cJSON *root;
    cJSON *state;
    int nparams;
    int rc;

    if (!m || !m->net || !m->net->pool || !m->net->pool->params || !json_path || !bin_path) {
        return -1;
    }

    root = sonn_to_json(m->net);
    state = som_state_json(m);
    if (!root || !state) {
        cJSON_Delete(root);
        cJSON_Delete(state);
        return -1;
    }
    cJSON_AddItemToObject(root, "som", state);

    rc = iojson_write_cjson(json_path, root);
    cJSON_Delete(root);
    if (rc != 0) return -1;

    nparams = m->net->pool->max_neurons * (m->net->pool->input_dim + 1);
    return iobin_write_floats(bin_path, m->net->pool->params, nparams);
}

som_t *som_load(const char *json_path, const char *bin_path, sonn_t **net_out)
{
    cJSON *root;
    cJSON *sj;
    cJSON *units;
    sonn_t *s;
    som_t *m;
    int *ids = NULL;
    int nparams;
    int rows = 0, cols = 0, n_steps = SOM_DEFAULT_N_STEPS, observe_count = 0;
    float eps_start = SOM_DEFAULT_EPS_START, eps_end = SOM_DEFAULT_EPS_END;
    float sigma_start = 0.0f, sigma_end = SOM_DEFAULT_SIGMA_END;

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

    sj = iojson_get_object(root, "som");
    units = sj ? iojson_get_array(sj, "units") : NULL;
    if (!sj || !units ||
        iojson_get_int(sj, "rows", &rows) != 0 ||
        iojson_get_int(sj, "cols", &cols) != 0) {
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    iojson_get_int(sj, "n_steps", &n_steps);
    iojson_get_int(sj, "observe_count", &observe_count);
    iojson_get_float(sj, "eps_start", &eps_start);
    iojson_get_float(sj, "eps_end", &eps_end);
    iojson_get_float(sj, "sigma_start", &sigma_start);
    iojson_get_float(sj, "sigma_end", &sigma_end);

    m = som_alloc(s, rows, cols);
    if (!m || cJSON_GetArraySize(units) != m->n_units) {
        som_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    /* Assign saved numbers directly. som_apply_schedule would replace a
     * non-positive sigma_start with the map default and hide a bad file. */
    m->n_steps = n_steps;
    m->observe_count = observe_count;
    m->eps_start = eps_start;
    m->eps_end = eps_end;
    m->sigma_start = sigma_start;
    m->sigma_end = sigma_end;

    ids = (int *)malloc((size_t)m->n_units * sizeof(int));
    if (!ids) {
        som_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }
    for (int k = 0; k < m->n_units; k++) {
        cJSON *item = cJSON_GetArrayItem(units, k);
        if (!cJSON_IsNumber(item)) {
            free(ids);
            som_destroy(m);
            sonn_destroy(s);
            cJSON_Delete(root);
            return NULL;
        }
        ids[k] = (int)item->valuedouble;
    }
    if (som_bind_units(m, ids) != 0) {
        free(ids);
        som_destroy(m);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    free(ids);
    cJSON_Delete(root);
    *net_out = s;
    return m;
}
