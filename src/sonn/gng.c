/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/gng.h>
#include <serialcore/iobin.h>
#include <serialcore/iojson.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float gng_sqdist(const float *prototype, const float *x, int dim, void *userdata)
{
    float dist = 0.0f;
    (void)userdata;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - prototype[i];
        dist += d * d;
    }
    return dist;
}

static int count_interior(const sonn_t *s)
{
    return s->current_neurons - s->in_count - s->out_count;
}

static float *age_at(gng_t *g, int id, int slot)
{
    size_t md = (size_t)g->net->pool->max_degree;
    return g->edge_age + (size_t)id * md + (size_t)slot;
}

static int gng_set_edge_age(gng_t *g, int from, int to, float age)
{
    nnpool_t *p = g->net->pool;
    int s1 = nnpool_find_edge_slot(p, from, to);
    int s2 = nnpool_find_edge_slot(p, to, from);
    if (s1 < 0 || s2 < 0) return -1;
    *age_at(g, from, s1) = age;
    *age_at(g, to, s2) = age;
    return 0;
}

/* Drop the undirected edge and zero both age slots. */
static void gng_unlink(gng_t *g, int a, int b)
{
    nnpool_t *p = g->net->pool;
    int s1 = nnpool_find_edge_slot(p, a, b);
    int s2 = nnpool_find_edge_slot(p, b, a);
    sonn_remove_edge(g->net, a, b);
    if (s1 >= 0) *age_at(g, a, s1) = 0.0f;
    if (s2 >= 0) *age_at(g, b, s2) = 0.0f;
}

/* Create the edge when missing, then store one age on both halves. */
static int gng_link(gng_t *g, int a, int b, float age)
{
    if (nnpool_find_edge_slot(g->net->pool, a, b) < 0) {
        if (sonn_add_edge(g->net, a, b) < 0) return -1;
    }
    return gng_set_edge_age(g, a, b, age);
}

static void gng_reset_unit_state(gng_t *g, int id)
{
    sonn_t *s = g->net;
    int md = s->pool->max_degree;
    int od = s->output_dim;

    g->error[id] = 0.0f;
    memset(age_at(g, id, 0), 0, (size_t)md * sizeof(float));
    if (od > 0) {
        memset(g->readout + (size_t)id * (size_t)od, 0, (size_t)od * sizeof(float));
    }
}

/* Zero this unit's algorithm state, including the age of every edge that still touches it. */
static void gng_prepare_forget(gng_t *g, int id)
{
    nnpool_t *p = g->net->pool;
    int md = p->max_degree;
    edge_t *row = nnpool_edge_row(p, id);

    if (row) {
        for (int j = 0; j < md; j++) {
            if (!row[j].active) continue;
            int to = row[j].to;
            int rj = nnpool_find_edge_slot(p, to, id);
            *age_at(g, id, j) = 0.0f;
            if (rj >= 0) *age_at(g, to, rj) = 0.0f;
        }
    }
    gng_reset_unit_state(g, id);
}

gng_t *gng_create(sonn_t *net, int insert_interval, float max_age, float error_decay)
{
    if (!net || !net->pool) return NULL;

    size_t max_n = (size_t)net->pool->max_neurons;
    size_t md = (size_t)net->pool->max_degree;
    size_t od = (size_t)net->output_dim;
    if (md != 0 && max_n > SIZE_MAX / md) return NULL;
    if (od != 0 && max_n > SIZE_MAX / od) return NULL;

    gng_t *g = (gng_t *)calloc(1, sizeof(gng_t));
    if (!g) return NULL;

    g->net = net;
    g->insert_interval = insert_interval < 0 ? GNG_DEFAULT_INSERT_INTERVAL : insert_interval;
    g->max_age = max_age > 0.0f ? max_age : GNG_DEFAULT_MAX_AGE;
    g->error_decay = error_decay > 0.0f ? error_decay : GNG_DEFAULT_ERROR_DECAY;
    g->distance = gng_sqdist;
    g->distance_userdata = NULL;

    g->error = (float *)calloc(max_n, sizeof(float));
    g->edge_age = (float *)calloc(max_n * md, sizeof(float));
    g->readout = (float *)calloc(max_n * od, sizeof(float));
    if (!g->error || !g->edge_age || (od > 0 && !g->readout)) {
        gng_destroy(g);
        return NULL;
    }
    return g;
}

void gng_destroy(gng_t *g)
{
    if (!g) return;
    free(g->error);
    free(g->edge_age);
    free(g->readout);
    free(g);
}

void gng_configure(gng_t *g, int insert_interval, float max_age, float error_decay)
{
    if (!g) return;
    if (insert_interval < 0) g->insert_interval = GNG_DEFAULT_INSERT_INTERVAL;
    else                     g->insert_interval = insert_interval;
    if (max_age > 0.0f)      g->max_age = max_age;
    else                     g->max_age = GNG_DEFAULT_MAX_AGE;
    if (error_decay > 0.0f)  g->error_decay = error_decay;
    else                     g->error_decay = GNG_DEFAULT_ERROR_DECAY;
}

void gng_set_distance(gng_t *g, gng_distance_fn fn, void *userdata)
{
    if (!g) return;
    g->distance = fn ? fn : gng_sqdist;
    g->distance_userdata = fn ? userdata : NULL;
}

float gng_prototype_distance(gng_t *g, int neuron_id, const float *input)
{
    if (!g || !g->net || !g->net->pool || !input || neuron_id < 0) return -1.0f;

    neuron_t *n = nnpool_get_neuron(g->net->pool, neuron_id);
    if (!n || !n->active || !n->weights) return -1.0f;

    gng_distance_fn fn = g->distance ? g->distance : gng_sqdist;
    return fn(n->weights, input, n->input_dim, g->distance_userdata);
}

int gng_find_bmu2(gng_t *g, const float *input, int *second_bmu)
{
    if (second_bmu) *second_bmu = -1;
    if (!g || !g->net || !g->net->pool || !input) return -1;

    int best = -1;
    int second = -1;
    float best_dist = 1e30f;
    float second_dist = 1e30f;
    sonn_t *s = g->net;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!sonn_is_interior(s, i)) continue;
        float dist = gng_prototype_distance(g, i, input);
        if (dist < 0.0f) continue;
        if (dist < best_dist) {
            second = best;
            second_dist = best_dist;
            best = i;
            best_dist = dist;
        } else if (dist < second_dist) {
            second = i;
            second_dist = dist;
        }
    }

    if (second_bmu) *second_bmu = second;
    return best;
}

int gng_find_bmu(gng_t *g, const float *input)
{
    return gng_find_bmu2(g, input, NULL);
}

void gng_adapt_prototype(sonn_t *s, int neuron_id, const float *input, float epsilon)
{
    if (!s || !s->pool || !input || neuron_id < 0 || epsilon <= 0.0f) return;

    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (!n || !n->active || !n->weights) return;

    for (int i = 0; i < n->input_dim; i++) {
        n->weights[i] += epsilon * (input[i] - n->weights[i]);
    }
}

typedef struct adapt_nb_ctx {
    sonn_t     *s;
    const float *x;
    float       eps;
} adapt_nb_ctx;

static int adapt_nb(int nb, void *userdata)
{
    adapt_nb_ctx *c = (adapt_nb_ctx *)userdata;
    if (c->eps > 0.0f && nb >= 0 && sonn_is_interior(c->s, nb)) {
        gng_adapt_prototype(c->s, nb, c->x, c->eps);
    }
    return 0;
}

void gng_adapt_bmu_and_neighbors(sonn_t *s, int bmu, const float *input, float epsilon_bmu, float epsilon_n)
{
    if (!s || !input || bmu < 0 || epsilon_bmu <= 0.0f) return;

    gng_adapt_prototype(s, bmu, input, epsilon_bmu);
    if (epsilon_n <= 0.0f) return;

    adapt_nb_ctx ctx;
    ctx.s = s;
    ctx.x = input;
    ctx.eps = epsilon_n;
    sonn_foreach_neighbor(s, bmu, adapt_nb, &ctx);
}

void gng_accumulate_error(gng_t *g, int neuron_id, float err)
{
    if (!g || !g->net || !g->error || neuron_id < 0 || neuron_id >= g->net->max_neurons) return;
    neuron_t *n = nnpool_get_neuron(g->net->pool, neuron_id);
    if (n && n->active) g->error[neuron_id] += err;
}

float gng_get_error(gng_t *g, int neuron_id)
{
    if (!g || !g->net || !g->error || neuron_id < 0 || neuron_id >= g->net->max_neurons) return 0.0f;
    neuron_t *n = nnpool_get_neuron(g->net->pool, neuron_id);
    if (n && n->active) return g->error[neuron_id];
    return 0.0f;
}

int gng_find_highest_error(gng_t *g)
{
    if (!g || !g->net || !g->net->pool || !g->error) return -1;

    int best = -1;
    float best_err = -1.0f;
    sonn_t *s = g->net;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!sonn_is_interior(s, i)) continue;
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (n && n->active && g->error[i] > best_err) {
            best_err = g->error[i];
            best = i;
        }
    }
    return best;
}

void gng_decay_errors(gng_t *g, float factor)
{
    if (!g || !g->net || !g->net->pool || !g->error) return;
    if (factor < 0.0f) factor = 0.0f;

    sonn_t *s = g->net;
    for (int i = 0; i < s->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (n && n->active && sonn_is_interior(s, i)) {
            g->error[i] *= factor;
        }
    }
}

static void age_undirected(gng_t *g, int from, int slot, int to)
{
    nnpool_t *p = g->net->pool;
    int rj = nnpool_find_edge_slot(p, to, from);
    *age_at(g, from, slot) += 1.0f;
    if (rj >= 0) *age_at(g, to, rj) += 1.0f;
}

void gng_age_edges(gng_t *g, int neuron_id)
{
    if (!g || !g->net || !g->net->pool || !g->edge_age) return;

    nnpool_t *p = g->net->pool;
    int md = p->max_degree;

    if (neuron_id >= 0) {
        neuron_t *n = nnpool_get_neuron(p, neuron_id);
        edge_t *row = nnpool_edge_row(p, neuron_id);
        if (!n || !n->active || !row) return;
        for (int j = 0; j < md; j++) {
            if (row[j].active) age_undirected(g, neuron_id, j, row[j].to);
        }
        return;
    }

    /* One logical age step per undirected edge, not one per stored half. */
    for (int i = 0; i < p->max_neurons; i++) {
        edge_t *row = nnpool_edge_row(p, i);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            if (!row[j].active || row[j].to <= i) continue;
            age_undirected(g, i, j, row[j].to);
        }
    }
}

void gng_reset_edge_age(gng_t *g, int from, int to)
{
    if (!g || !g->net || !g->net->pool) return;
    gng_set_edge_age(g, from, to, 0.0f);
}

float gng_get_edge_age(gng_t *g, int from, int to)
{
    if (!g || !g->net || !g->net->pool || !g->edge_age) return -1.0f;
    int slot = nnpool_find_edge_slot(g->net->pool, from, to);
    if (slot < 0) return -1.0f;
    return *age_at(g, from, slot);
}

int gng_remove_old_edges(gng_t *g, float max_age)
{
    if (!g || !g->net || !g->net->pool || !g->edge_age || max_age <= 0.0f) return 0;

    nnpool_t *p = g->net->pool;
    int md = p->max_degree;
    int removed = 0;

    for (int i = 0; i < p->max_neurons; i++) {
        edge_t *row = nnpool_edge_row(p, i);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            if (!row[j].active || row[j].to <= i) continue;
            if (*age_at(g, i, j) <= max_age) continue;
            gng_unlink(g, i, row[j].to);
            removed++;
        }
    }
    return removed;
}

int gng_insert_between(gng_t *g, int a, int b, activaton_t type)
{
    if (!g || !g->net || !g->net->pool || a < 0 || b < 0) return -1;

    sonn_t *s = g->net;
    neuron_t *na = nnpool_get_neuron(s->pool, a);
    neuron_t *nb = nnpool_get_neuron(s->pool, b);
    if (!na || !nb || !na->active || !nb->active || !na->weights || !nb->weights) return -1;

    int had = nnpool_find_edge_slot(s->pool, a, b) >= 0;
    float saved_age = had ? gng_get_edge_age(g, a, b) : 0.0f;
    float err_a = g->error[a];
    float err_b = g->error[b];

    int new_id = sonn_add_neuron(s, type);
    if (new_id < 0) return -1;

    neuron_t *nn = nnpool_get_neuron(s->pool, new_id);
    if (!nn || !nn->weights) {
        gng_prepare_forget(g, new_id);
        sonn_remove_neuron(s, new_id);
        return -1;
    }

    gng_reset_unit_state(g, new_id);
    for (int i = 0; i < nn->input_dim; i++) {
        float va = (i < na->input_dim) ? na->weights[i] : 0.0f;
        float vb = (i < nb->input_dim) ? nb->weights[i] : 0.0f;
        nn->weights[i] = 0.5f * (va + vb);
    }

    if (had) gng_unlink(g, a, b);

    if (gng_link(g, a, new_id, 0.0f) != 0 || gng_link(g, b, new_id, 0.0f) != 0) {
        gng_prepare_forget(g, new_id);
        sonn_remove_neuron(s, new_id);
        if (had) gng_link(g, a, b, saved_age);
        return -1;
    }

    g->error[new_id] = 0.5f * err_a;
    g->error[a] = 0.5f * err_a;
    g->error[b] = 0.5f * err_b;

    int od = s->output_dim;
    if (od > 0 && g->readout) {
        float *ra = g->readout + (size_t)a * (size_t)od;
        float *rb = g->readout + (size_t)b * (size_t)od;
        float *rn = g->readout + (size_t)new_id * (size_t)od;
        for (int i = 0; i < od; i++) rn[i] = 0.5f * (ra[i] + rb[i]);
    }
    return new_id;
}

static int seed_interior(gng_t *g, const float *x)
{
    sonn_t *s = g->net;
    neuron_t *anchor = nnpool_get_neuron(s->pool, s->in_start);
    activaton_t type = (anchor && anchor->active) ? anchor->type : GELU;

    int id = sonn_add_neuron(s, type);
    if (id < 0) return -1;

    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->weights) {
        gng_prepare_forget(g, id);
        sonn_remove_neuron(s, id);
        return -1;
    }
    for (int i = 0; i < n->input_dim; i++) n->weights[i] = x[i];
    gng_reset_unit_state(g, id);
    return id;
}

typedef struct heb_ctx {
    gng_t *g;
    int    best;
    float  best_err;
} heb_ctx;

static int consider_neighbor(int nb, void *userdata)
{
    heb_ctx *c = (heb_ctx *)userdata;
    sonn_t *s = c->g->net;
    neuron_t *n = nnpool_get_neuron(s->pool, nb);
    if (!n || !n->active || !sonn_is_interior(s, nb)) return 0;
    if (c->g->error[nb] > c->best_err) {
        c->best_err = c->g->error[nb];
        c->best = nb;
    }
    return 0;
}

static int highest_error_neighbor(gng_t *g, int id)
{
    heb_ctx ctx;
    ctx.g = g;
    ctx.best = -1;
    ctx.best_err = -1.0f;
    sonn_foreach_neighbor(g->net, id, consider_neighbor, &ctx);
    return ctx.best;
}

/* Keep at least two interior units so a fresh seed pair is not deleted. */
static int gng_remove_isolated(gng_t *g)
{
    sonn_t *s = g->net;
    int removed = 0;
    int guard = s->pool->max_neurons;

    for (;;) {
        if (count_interior(s) <= 2) break;
        int victim = -1;
        for (int i = 0; i < s->pool->max_neurons; i++) {
            neuron_t *n = nnpool_get_neuron(s->pool, i);
            if (!n || !n->active || !sonn_is_interior(s, i)) continue;
            if (s->pool->degrees[i] == 0) {
                victim = i;
                break;
            }
        }
        if (victim < 0) break;
        gng_prepare_forget(g, victim);
        sonn_remove_neuron(s, victim);
        removed++;
        if (--guard < 0) break;
    }
    return removed;
}

static void update_readout(gng_t *g, int id, const float *y, float eps_out)
{
    if (!y || eps_out <= 0.0f || id < 0 || !g->readout) return;
    sonn_t *s = g->net;
    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->active) return;

    int od = s->output_dim;
    float *r = g->readout + (size_t)id * (size_t)od;
    for (int i = 0; i < od; i++) {
        r[i] += eps_out * (y[i] - r[i]);
    }
}

int gng_observe(gng_t *g, const float *x, const float *y, float eps_bmu, float eps_n, float eps_out)
{
    if (!g || !g->net || !g->net->pool || !x) return -1;
    sonn_t *s = g->net;

    /* Two connected seeds. A single unit has no neighbor, so insertion cannot start. */
    if (count_interior(s) == 0) {
        int id = seed_interior(g, x);
        if (id < 0) return -1;
        update_readout(g, id, y, eps_out);
        g->observe_count++;
        return 0;
    }
    if (count_interior(s) == 1) {
        int first = gng_find_bmu(g, x);
        int second = seed_interior(g, x);
        if (second < 0) return -1;
        if (first < 0 || gng_link(g, first, second, 0.0f) != 0) {
            gng_prepare_forget(g, second);
            sonn_remove_neuron(s, second);
            return -1;
        }
        update_readout(g, second, y, eps_out);
        g->observe_count++;
        return 0;
    }

    /* 1. BMU and second BMU. */
    int second = -1;
    int bmu = gng_find_bmu2(g, x, &second);
    if (bmu < 0) return -1;

    /* 2. Age the BMU's edges before the competitive-Hebbian refresh. */
    gng_age_edges(g, bmu);

    /* 3. Error uses the distance from before the prototype moves. */
    float dist = gng_prototype_distance(g, bmu, x);
    if (dist >= 0.0f) gng_accumulate_error(g, bmu, dist);

    /* 4. Adapt BMU and interior neighbors. */
    gng_adapt_bmu_and_neighbors(s, bmu, x, eps_bmu, eps_n);

    /* 5. Link BMU to second BMU and leave that edge at age 0. */
    if (second >= 0) gng_link(g, bmu, second, 0.0f);

    /* 6. Prune old edges, then drop isolated units while at least two remain. */
    gng_remove_old_edges(g, g->max_age);
    gng_remove_isolated(g);

    /* 7. Insert, then decay every error (including the new unit). */
    g->observe_count++;
    if (g->insert_interval > 0 && (g->observe_count % g->insert_interval) == 0) {
        int q = gng_find_highest_error(g);
        if (q >= 0) {
            int f = highest_error_neighbor(g, q);
            if (f >= 0) {
                neuron_t *qn = nnpool_get_neuron(s->pool, q);
                activaton_t type = (qn && qn->active) ? qn->type : GELU;
                gng_insert_between(g, q, f, type);
            }
        }
    }
    gng_decay_errors(g, g->error_decay);

    /* 8. Local readout. A global output-anchor average is not a mapping. */
    neuron_t *bn = nnpool_get_neuron(s->pool, bmu);
    if (bn && bn->active) update_readout(g, bmu, y, eps_out);
    return 0;
}

int gng_predict(gng_t *g, const float *x, float *out)
{
    if (!g || !g->net || !x || !out) return -1;

    int od = g->net->output_dim;
    for (int i = 0; i < od; i++) out[i] = 0.0f;

    int bmu = gng_find_bmu(g, x);
    if (bmu < 0 || !g->readout) return -1;

    const float *r = g->readout + (size_t)bmu * (size_t)od;
    for (int i = 0; i < od; i++) out[i] = r[i];
    return 0;
}

static cJSON *gng_state_json(const gng_t *g)
{
    sonn_t *s = g->net;
    cJSON *gj = cJSON_CreateObject();
    cJSON *units = cJSON_CreateArray();
    cJSON *edges = cJSON_CreateArray();
    if (!gj || !units || !edges) {
        cJSON_Delete(gj);
        cJSON_Delete(units);
        cJSON_Delete(edges);
        return NULL;
    }

    cJSON_AddNumberToObject(gj, "insert_interval", g->insert_interval);
    cJSON_AddNumberToObject(gj, "max_age", g->max_age);
    cJSON_AddNumberToObject(gj, "error_decay", g->error_decay);
    cJSON_AddNumberToObject(gj, "observe_count", g->observe_count);
    cJSON_AddItemToObject(gj, "units", units);
    cJSON_AddItemToObject(gj, "edges", edges);

    int od = s->output_dim;
    int md = s->pool->max_degree;
    for (int id = 0; id < s->pool->max_neurons; id++) {
        neuron_t *n = nnpool_get_neuron(s->pool, id);
        edge_t *row;
        cJSON *ju;
        cJSON *rv;

        if (!n || !n->active || !sonn_is_interior(s, id)) continue;

        ju = cJSON_CreateObject();
        rv = iojson_floats_to_array(g->readout + (size_t)id * (size_t)od, od);
        if (!ju || !rv) {
            cJSON_Delete(ju);
            cJSON_Delete(rv);
            cJSON_Delete(gj);
            return NULL;
        }
        cJSON_AddNumberToObject(ju, "id", id);
        cJSON_AddNumberToObject(ju, "error", g->error[id]);
        cJSON_AddItemToObject(ju, "readout", rv);
        cJSON_AddItemToArray(units, ju);

        row = nnpool_edge_row(s->pool, id);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            cJSON *je;
            if (!row[j].active || row[j].to <= id) continue;
            je = cJSON_CreateObject();
            if (!je) {
                cJSON_Delete(gj);
                return NULL;
            }
            cJSON_AddNumberToObject(je, "from", id);
            cJSON_AddNumberToObject(je, "to", row[j].to);
            cJSON_AddNumberToObject(je, "age", g->edge_age[(size_t)id * (size_t)md + (size_t)j]);
            cJSON_AddItemToArray(edges, je);
        }
    }
    return gj;
}

static int gng_import_state(gng_t *g, const cJSON *gj)
{
    sonn_t *s = g->net;
    cJSON *units;
    cJSON *edges;
    int observe_count = 0;
    int od;

    if (!gj) return 0;
    if (iojson_get_int(gj, "observe_count", &observe_count) == 0) {
        g->observe_count = observe_count;
    }

    od = s->output_dim;
    units = iojson_get_array(gj, "units");
    if (units) {
        int n = cJSON_GetArraySize(units);
        for (int i = 0; i < n; i++) {
            cJSON *ju = cJSON_GetArrayItem(units, i);
            cJSON *rv;
            int id = 0;
            float err = 0.0f;
            neuron_t *neu;

            if (!cJSON_IsObject(ju) ||
                iojson_get_int(ju, "id", &id) != 0 ||
                iojson_get_float(ju, "error", &err) != 0) {
                return -1;
            }
            if (id < 0 || id >= s->max_neurons) return -1;
            neu = nnpool_get_neuron(s->pool, id);
            if (!neu || !neu->active || !sonn_is_interior(s, id)) return -1;

            g->error[id] = err;
            rv = iojson_get_array(ju, "readout");
            if (!rv || iojson_array_to_floats(rv, g->readout + (size_t)id * (size_t)od, od) != 0) {
                return -1;
            }
        }
    }

    edges = iojson_get_array(gj, "edges");
    if (edges) {
        int n = cJSON_GetArraySize(edges);
        for (int i = 0; i < n; i++) {
            cJSON *je = cJSON_GetArrayItem(edges, i);
            int from = 0, to = 0;
            float age = 0.0f;

            if (!cJSON_IsObject(je) ||
                iojson_get_int(je, "from", &from) != 0 ||
                iojson_get_int(je, "to", &to) != 0 ||
                iojson_get_float(je, "age", &age) != 0) {
                return -1;
            }
            if (gng_set_edge_age(g, from, to, age) != 0) return -1;
        }
    }
    return 0;
}

int gng_save(const gng_t *g, const char *json_path, const char *bin_path)
{
    cJSON *root;
    cJSON *state;
    int nparams;
    int rc;

    if (!g || !g->net || !g->net->pool || !g->net->pool->params || !json_path || !bin_path) {
        return -1;
    }

    root = sonn_to_json(g->net);
    state = gng_state_json(g);
    if (!root || !state) {
        cJSON_Delete(root);
        cJSON_Delete(state);
        return -1;
    }
    cJSON_AddItemToObject(root, "gng", state);

    rc = iojson_write_cjson(json_path, root);
    cJSON_Delete(root);
    if (rc != 0) return -1;

    nparams = g->net->pool->max_neurons * (g->net->pool->input_dim + 1);
    return iobin_write_floats(bin_path, g->net->pool->params, nparams);
}

gng_t *gng_load(const char *json_path, const char *bin_path, sonn_t **net_out)
{
    cJSON *root;
    cJSON *gj;
    sonn_t *s;
    gng_t *g;
    int nparams;
    int interval = GNG_DEFAULT_INSERT_INTERVAL;
    float max_age = GNG_DEFAULT_MAX_AGE;
    float decay = GNG_DEFAULT_ERROR_DECAY;

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

    gj = iojson_get_object(root, "gng");
    if (gj) {
        iojson_get_int(gj, "insert_interval", &interval);
        iojson_get_float(gj, "max_age", &max_age);
        iojson_get_float(gj, "error_decay", &decay);
    }

    g = gng_create(s, interval, max_age, decay);
    if (!g || (gj && gng_import_state(g, gj) != 0)) {
        gng_destroy(g);
        sonn_destroy(s);
        cJSON_Delete(root);
        return NULL;
    }

    cJSON_Delete(root);
    *net_out = s;
    return g;
}
