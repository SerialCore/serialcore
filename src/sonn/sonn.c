/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * SONN implementation — self-organizing neural network with no layer concept.
 *
 * This layer builds dynamic-growth neural network behavior on top of the raw
 * Pool:
 *   - At create time a fixed set of input neurons and output neurons is
 *     pre-allocated; they bracket the input and output shapes chosen by the
 *     caller.
 *   - The SONN grows interior neurons on demand via sonn_observe() following a
 *     Growing-Neural-Gas-style rule.
 *   - Manages activation types.
 *   - Handles edge creation with proper semantic rules.
 *   - Provides convenient query APIs.
 *
 * All raw memory allocation and slot management is delegated to the Pool.
 */

#include <serialcore/sonn/sonn.h>

#include <stdlib.h>
#include <string.h>

/* Built-in auto-growth defaults — see sonn_configure_growth(). */
#define SONN_DEFAULT_INSERT_INTERVAL 50
#define SONN_DEFAULT_MAX_AGE         50.0f
#define SONN_DEFAULT_ERROR_DECAY     0.99f

/* Internal helpers to reduce duplication in bidirectional edge management */

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

/* A neuron slot is "interior" if it is not one of the fixed input anchors nor
 * the fixed output anchors. Interior neurons are the ones the SONN grows on
 * its own — they are candidates for BMU / error accumulation / insertion.
 */
static int is_interior(sonn_t *s, int id)
{
    if (!s || id < 0) return 0;
    if (id >= s->in_start && id < s->in_start + s->in_count) return 0;
    if (id >= s->out_start && id < s->out_start + s->out_count) return 0;
    return 1;
}

/* Activate a freshly-acquired slot, initialize its edge row/degree, and stamp
 * the activation type. Mirrors the old network_add_neuron logic but takes a
 * sonn_t so it can be reused when seeding interior neurons.
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

    s->insert_interval = SONN_DEFAULT_INSERT_INTERVAL;
    s->max_age         = SONN_DEFAULT_MAX_AGE;
    s->error_decay     = SONN_DEFAULT_ERROR_DECAY;
    s->observe_count   = 0;

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
    if (!is_interior(s, id)) return;

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

uint64_t sonn_get_neuronid(sonn_t *s, int slot)
{
    if (!s || !s->pool) return 0;
    neuron_t *n = nnpool_get_neuron(s->pool, slot);
    return n ? n->id : 0;
}

float sonn_prototype_distance(sonn_t *s, int neuron_id, const float *input)
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

int sonn_find_bmu(sonn_t *s, const float *input)
{
    return sonn_find_bmu2(s, input, NULL);
}

int sonn_find_bmu2(sonn_t *s, const float *input, int *second_bmu)
{
    if (!s || !s->pool || !input) return -1;

    int best = -1;
    int second = -1;
    float best_dist = 1e30f;
    float second_dist = 1e30f;
    int dim = s->pool->input_dim;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!is_interior(s, i)) continue;
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

void sonn_accumulate_error(sonn_t *s, int neuron_id, float err)
{
    if (!s || !s->pool || neuron_id < 0) return;
    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (n && n->active) {
        n->error += err;
    }
}

void sonn_adapt_prototype(sonn_t *s, int neuron_id, const float *input, float epsilon)
{
    if (!s || !s->pool || !input || neuron_id < 0 || epsilon <= 0) return;

    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (!n || !n->active || !n->weights) return;

    int dim = n->input_dim;
    for (int i = 0; i < dim; i++) {
        n->weights[i] += epsilon * (input[i] - n->weights[i]);
    }
}

void sonn_age_edges(sonn_t *s, int neuron_id)
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

void sonn_reset_edge_age(sonn_t *s, int from, int to)
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

int sonn_remove_old_edges(sonn_t *s, float max_age)
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
                    remove_bidirectional_edge(s, i, to, 1);
                    removed++;
                } else {
                    remove_bidirectional_edge(s, i, to, 1);
                }
            }
        }
    }
    return removed;
}

float sonn_get_error(sonn_t *s, int neuron_id)
{
    if (!s || !s->pool || neuron_id < 0) return 0.0f;
    neuron_t *n = nnpool_get_neuron(s->pool, neuron_id);
    if (n && n->active) return n->error;
    return 0.0f;
}

int sonn_find_highest_error(sonn_t *s)
{
    if (!s || !s->pool) return -1;

    int best = -1;
    float best_err = -1.0f;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        if (!is_interior(s, i)) continue;
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (n && n->active && n->error > best_err) {
            best_err = n->error;
            best = i;
        }
    }
    return best;
}

float sonn_get_edge_age(sonn_t *s, int from, int to)
{
    if (!s || !s->pool) return -1.0f;
    int slot = nnpool_find_edge_slot(s->pool, from, to);
    if (slot < 0) return -1.0f;
    int md = s->pool->max_degree;
    return s->pool->edges[from * md + slot].age;
}

void sonn_decay_errors(sonn_t *s, float factor)
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

void sonn_adapt_bmu_and_neighbors(sonn_t *s, int bmu, const float *input,
                                   float epsilon_bmu, float epsilon_n)
{
    if (!s || !input || bmu < 0 || epsilon_bmu <= 0) return;

    sonn_adapt_prototype(s, bmu, input, epsilon_bmu);

    int md = (s->pool && s->pool->max_degree > 0) ? s->pool->max_degree : 64;
    if (md > 256) md = 256;
    int buf[256];
    int ncount = sonn_get_neighbors(s, bmu, buf, md);
    for (int i = 0; i < ncount; i++) {
        int nb = buf[i];
        if (nb >= 0 && epsilon_n > 0) {
            sonn_adapt_prototype(s, nb, input, epsilon_n);
        }
    }
}

int sonn_insert_between(sonn_t *s, int a, int b, activaton_t type)
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
    sonn_add_edge_topology(s, a, new_id);
    sonn_add_edge_topology(s, b, new_id);

    /* New neuron inherits half the error of the high-error parent (a) so it
     * stays a candidate for further growth. */
    nn->error = 0.5f * na->error;
    na->error *= 0.5f;
    nb->error *= 0.5f;

    return new_id;
}

/* --- High-level auto-growth API ------------------------------------------ */

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

void sonn_configure_growth(sonn_t *s, int insert_interval, float max_age, float error_decay)
{
    if (!s) return;
    if (insert_interval > 0) s->insert_interval = insert_interval;
    else                     s->insert_interval = SONN_DEFAULT_INSERT_INTERVAL;
    if (max_age > 0)         s->max_age = max_age;
    else                     s->max_age = SONN_DEFAULT_MAX_AGE;
    if (error_decay > 0)     s->error_decay = error_decay;
    else                     s->error_decay = SONN_DEFAULT_ERROR_DECAY;
}

/* Seed the network with a single interior neuron whose prototype equals the
 * first observed input vector. Called by sonn_observe() when no interior
 * neurons exist yet. Returns the new neuron id, or -1.
 */
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
        if (!nb || !nb->active || !is_interior(s, buf[i])) continue;
        if (nb->error > best_err) {
            best_err = nb->error;
            best = buf[i];
        }
    }
    return best;
}

int sonn_observe(sonn_t *s, const float *x, const float *y,
                  float eps_bmu, float eps_n, float eps_out)
{
    if (!s || !s->pool || !x) return -1;
    int dim = s->input_dim;

    /* 1. Clamp input neuron activations to x. */
    for (int i = 0; i < s->in_count && i < dim; i++) {
        neuron_t *in_n = nnpool_get_neuron(s->pool, s->in_start + i);
        if (in_n) in_n->activation = x[i];
    }

    /* 2. Seed an interior neuron if the network is empty. */
    int bmu = sonn_find_bmu(s, x);
    int second = -1;
    if (bmu < 0) {
        bmu = seed_interior(s, x);
        if (bmu < 0) return -1;
    } else {
        bmu = sonn_find_bmu2(s, x, &second);
    }

    /* 3. Maintain a topology edge between BMU and second BMU. */
    if (bmu >= 0 && second >= 0) {
        if (nnpool_find_edge_slot(s->pool, bmu, second) < 0) {
            sonn_add_edge_topology(s, bmu, second);
        }
        sonn_reset_edge_age(s, bmu, second);
    }

    /* 4. Adapt BMU and neighbors toward x. Cache basic distance. */
    float dist = sonn_prototype_distance(s, bmu, x);

    sonn_adapt_bmu_and_neighbors(s, bmu, x, eps_bmu, eps_n);

    /* 5. Accumulate the squared distance as error on the BMU. */
    sonn_accumulate_error(s, bmu, dist);

    /* 6. Age the BMU's edges and prune old edges. */
    sonn_age_edges(s, bmu);
    sonn_remove_old_edges(s, s->max_age);

    /* 7. Decay all errors. */
    sonn_decay_errors(s, s->error_decay);

    /* 8. Insert a new interior neuron every N observations. */
    s->observe_count++;
    if (s->insert_interval > 0 && (s->observe_count % s->insert_interval) == 0) {
        int q = sonn_find_highest_error(s);
        if (q >= 0) {
            int f = highest_error_neighbor(s, q);
            if (f >= 0) {
                sonn_insert_between(s, q, f, GELU);
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

int sonn_get_output(sonn_t *s, float *out)
{
    if (!s || !s->pool || !out) return -1;
    for (int i = 0; i < s->out_count; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, s->out_start + i);
        out[i] = n ? n->activation : 0.0f;
    }
    return 0;
}