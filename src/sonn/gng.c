/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/gng.h>

#include <stdlib.h>

gng_t *gng_create(sonn_t *net, int insert_interval, float max_age, float error_decay)
{
    if (!net) return NULL;

    gng_t *g = (gng_t *)calloc(1, sizeof(gng_t));
    if (!g) return NULL;

    g->net             = net;
    g->insert_interval = insert_interval > 0 ? insert_interval : GNG_DEFAULT_INSERT_INTERVAL;
    g->max_age         = max_age > 0       ? max_age        : GNG_DEFAULT_MAX_AGE;
    g->error_decay     = error_decay > 0   ? error_decay    : GNG_DEFAULT_ERROR_DECAY;
    g->observe_count   = 0;
    return g;
}

void gng_destroy(gng_t *g)
{
    free(g);
}

void gng_configure(gng_t *g, int insert_interval, float max_age, float error_decay)
{
    if (!g) return;
    if (insert_interval > 0) g->insert_interval = insert_interval;
    else                     g->insert_interval = GNG_DEFAULT_INSERT_INTERVAL;
    if (max_age > 0)         g->max_age = max_age;
    else                     g->max_age = GNG_DEFAULT_MAX_AGE;
    if (error_decay > 0)     g->error_decay = error_decay;
    else                     g->error_decay = GNG_DEFAULT_ERROR_DECAY;
}

float gng_prototype_distance(sonn_t *s, int neuron_id, const float *input)
{
    if (!s || !s->pool || !input || neuron_id < 0) return -1.0f;

    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (!n || !n->active || !n->weights) return -1.0f;

    float dist = 0.0f;
    int dim = n->input_dim;
    for (int i = 0; i < dim; i++) {
        float d = input[i] - n->weights[i];
        dist += d * d;
    }
    return dist;
}

int gng_find_bmu(sonn_t *s, const float *input)
{
    return gng_find_bmu2(s, input, NULL);
}

int gng_find_bmu2(sonn_t *s, const float *input, int *second_bmu)
{
    if (!s || !s->pool || !input) return -1;

    int best = -1;
    int second = -1;
    float best_dist = 1e30f;
    float second_dist = 1e30f;
    int dim = s->pool->input_dim;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!sonn_is_interior(s, i)) continue;
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (!n || !n->active || !n->weights) continue;

        float dist = 0.0f;
        for (int j = 0; j < dim; j++) {
            float d = input[j] - n->weights[j];
            dist += d * d;
        }
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

void gng_adapt_prototype(sonn_t *s, int neuron_id, const float *input, float epsilon)
{
    if (!s || !s->pool || !input || neuron_id < 0 || epsilon <= 0) return;

    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (!n || !n->active || !n->weights) return;

    int dim = n->input_dim;
    for (int i = 0; i < dim; i++) {
        n->weights[i] += epsilon * (input[i] - n->weights[i]);
    }
}

void gng_adapt_bmu_and_neighbors(sonn_t *s, int bmu, const float *input, float epsilon_bmu, float epsilon_n)
{
    if (!s || !input || bmu < 0 || epsilon_bmu <= 0) return;

    gng_adapt_prototype(s, bmu, input, epsilon_bmu);

    int md = (s->pool && s->pool->max_degree > 0) ? s->pool->max_degree : 64;
    if (md > 256) md = 256;
    int buf[256];
    int ncount = sonn_get_neighbors(s, bmu, buf, md);
    for (int i = 0; i < ncount; i++) {
        int nb = buf[i];
        if (nb >= 0 && epsilon_n > 0) {
            gng_adapt_prototype(s, nb, input, epsilon_n);
        }
    }
}

void gng_accumulate_error(sonn_t *s, int neuron_id, float err)
{
    if (!s || !s->pool || neuron_id < 0) return;
    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (n && n->active) {
        n->error += err;
    }
}

float gng_get_error(sonn_t *s, int neuron_id)
{
    if (!s || !s->pool || neuron_id < 0) return 0.0f;
    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (n && n->active) return n->error;
    return 0.0f;
}

int gng_find_highest_error(sonn_t *s)
{
    if (!s || !s->pool) return -1;

    int best = -1;
    float best_err = -1.0f;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!sonn_is_interior(s, i)) continue;
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (n && n->active && n->error > best_err) {
            best_err = n->error;
            best = i;
        }
    }
    return best;
}

void gng_decay_errors(sonn_t *s, float factor)
{
    if (!s || !s->pool) return;
    if (factor < 0) factor = 0;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (n && n->active) {
            n->error *= factor;
        }
    }
}

void gng_age_edges(sonn_t *s, int neuron_id)
{
    if (!s || !s->pool) return;

    int md = s->pool->max_degree;

    if (neuron_id >= 0) {
        edge_t *row = nnpool_edge_row(s->pool, neuron_id);
        if (!row) return;
        for (int i = 0; i < md; i++) {
            if (row[i].active) {
                row[i].age += 1.0f;
            }
        }
    } else {
        for (int i = 0; i < s->pool->max_neurons; i++) {
            edge_t *row = nnpool_edge_row(s->pool, i);
            if (!row) continue;
            for (int j = 0; j < md; j++) {
                if (row[j].active) {
                    row[j].age += 1.0f;
                }
            }
        }
    }
}

void gng_reset_edge_age(sonn_t *s, int from, int to)
{
    if (!s || !s->pool) return;

    int slot = nnpool_find_edge_slot(s->pool, from, to);
    if (slot >= 0) {
        int md = s->pool->max_degree;
        s->pool->edges[from * md + slot].age = 0.0f;
    }

    int rslot = nnpool_find_edge_slot(s->pool, to, from);
    if (rslot >= 0) {
        int md = s->pool->max_degree;
        s->pool->edges[to * md + rslot].age = 0.0f;
    }
}

float gng_get_edge_age(sonn_t *s, int from, int to)
{
    if (!s || !s->pool) return -1.0f;
    int slot = nnpool_find_edge_slot(s->pool, from, to);
    if (slot < 0) return -1.0f;
    int md = s->pool->max_degree;
    return s->pool->edges[from * md + slot].age;
}

int gng_remove_old_edges(sonn_t *s, float max_age)
{
    if (!s || !s->pool || max_age <= 0) return 0;

    int removed = 0;
    int md = s->pool->max_degree;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        edge_t *row = nnpool_edge_row(s->pool, i);
        if (!row) continue;

        for (int j = 0; j < md; j++) {
            if (row[j].active && row[j].age > max_age) {
                int to = row[j].to;
                if (to > i) {
                    sonn_remove_edge(s, i, to);
                    removed++;
                } else {
                    sonn_remove_edge(s, i, to);
                }
            }
        }
    }
    return removed;
}

int gng_insert_between(sonn_t *s, int a, int b, activaton_t type)
{
    if (!s || !s->pool || a < 0 || b < 0) return -1;

    neuron_t *na = nnpool_get_neuron(s->pool, a);
    neuron_t *nb = nnpool_get_neuron(s->pool, b);
    if (!na || !nb || !na->active || !nb->active) return -1;

    int new_id = sonn_add_neuron(s, type);
    if (new_id < 0) return -1;

    neuron_t *nn = nnpool_get_neuron(s->pool, new_id);
    if (!nn || !nn->weights) {
        sonn_remove_neuron(s, new_id);
        return -1;
    }

    int dim = nn->input_dim;
    for (int i = 0; i < dim; i++) {
        float va = (na->weights && i < na->input_dim) ? na->weights[i] : 0.0f;
        float vb = (nb->weights && i < nb->input_dim) ? nb->weights[i] : 0.0f;
        nn->weights[i] = 0.5f * (va + vb);
    }

    sonn_remove_edge(s, a, b);
    sonn_add_edge(s, a, new_id);
    sonn_add_edge(s, b, new_id);

    /* New neuron inherits half the error of the high-error parent (a) so it stays a candidate for further growth. */
    nn->error = 0.5f * na->error;
    na->error *= 0.5f;
    nb->error *= 0.5f;

    return new_id;
}

/* Seed the network with a single interior neuron whose prototype equals the first observed input vector. */
static int seed_interior(sonn_t *s, const float *x)
{
    int id = sonn_add_neuron(s, GELU);
    if (id < 0) return -1;
    neuron_t *n = nnpool_get_neuron(s->pool, id);
    if (!n || !n->weights) {
        sonn_remove_neuron(s, id);
        return -1;
    }
    int dim = s->input_dim;
    for (int i = 0; i < dim; i++) {
        n->weights[i] = x[i];
    }
    n->error = 0.0f;
    return id;
}

/* Find the highest-error neighbor of `a` and return its id (or -1). */
static int highest_error_neighbor(sonn_t *s, int a)
{
    int md = s->pool->max_degree;
    if (md > 256) md = 256;
    int buf[256];
    int n = sonn_get_neighbors(s, a, buf, md);

    int best = -1;
    float best_err = -1.0f;
    for (int i = 0; i < n; i++) {
        neuron_t *nb = nnpool_get_neuron(s->pool, buf[i]);
        if (!nb || !nb->active || !sonn_is_interior(s, buf[i])) continue;
        if (nb->error > best_err) {
            best_err = nb->error;
            best = buf[i];
        }
    }
    return best;
}

int gng_observe(gng_t *g, const float *x, const float *y, float eps_bmu, float eps_n, float eps_out)
{
    if (!g || !g->net || !g->net->pool || !x) return -1;
    sonn_t *s = g->net;
    int dim = s->input_dim;

    /* 1. Clamp input neuron activations to x. */
    for (int i = 0; i < s->in_count && i < dim; i++) {
        neuron_t *in_n = nnpool_get_neuron(s->pool, s->in_start + i);
        if (in_n) in_n->activation = x[i];
    }

    /* 2. Seed an interior neuron if the network is empty. */
    int bmu = gng_find_bmu(s, x);
    int second = -1;
    if (bmu < 0) {
        bmu = seed_interior(s, x);
        if (bmu < 0) return -1;
    } else {
        bmu = gng_find_bmu2(s, x, &second);
    }

    /* 3. Maintain a topology edge between BMU and second BMU. */
    if (bmu >= 0 && second >= 0) {
        if (nnpool_find_edge_slot(s->pool, bmu, second) < 0) {
            sonn_add_edge(s, bmu, second);
        }
        gng_reset_edge_age(s, bmu, second);
    }

    /* 4. Adapt BMU and neighbors toward x. Cache basic distance. */
    float dist = gng_prototype_distance(s, bmu, x);
    gng_adapt_bmu_and_neighbors(s, bmu, x, eps_bmu, eps_n);

    /* 5. Accumulate the squared distance as error on the BMU. */
    gng_accumulate_error(s, bmu, dist);

    /* 6. Age the BMU's edges and prune old edges. */
    gng_age_edges(s, bmu);
    gng_remove_old_edges(s, g->max_age);

    /* 7. Decay all errors. */
    gng_decay_errors(s, g->error_decay);

    /* 8. Insert a new interior neuron every N observations. */
    g->observe_count++;
    if (g->insert_interval > 0 && (g->observe_count % g->insert_interval) == 0) {
        int q = gng_find_highest_error(s);
        if (q >= 0) {
            int f = highest_error_neighbor(s, q);
            if (f >= 0) {
                gng_insert_between(s, q, f, GELU);
            }
        }
    }

    /* 9. If a supervised target is given, pull each output anchor's
     *    activation toward y[i]. Alternatively, when no y is provided, decay
     *    the output activations toward their prototype-supplied value.
     */
    if (y) {
        for (int i = 0; i < s->out_count; i++) {
            neuron_t *out_n = nnpool_get_neuron(s->pool, s->out_start + i);
            if (!out_n) continue;
            out_n->activation += eps_out * (y[i] - out_n->activation);
        }
    }

    return 0;
}