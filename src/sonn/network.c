/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * SONN Network implementation — high-level semantics.
 *
 * This layer builds meaningful neural network behavior on top of the raw Pool:
 *   - Tracks which neurons are "active" from the user's perspective.
 *   - Manages activation types.
 *   - Handles edge creation with proper semantic rules.
 *   - Provides convenient query APIs.
 *
 * All raw memory allocation and slot management is delegated to the Pool.
 */

#include <serialcore/sonn/network.h>

#include <stdlib.h>
#include <string.h>

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
 * write_weight != 0  → also store 'weight' in receiver's weights[rslot] (Option A, for FFNN)
 * write_weight == 0  → pure topology (for GNG / self-organizing)
 */
static int add_bidirectional_edge(network_t *net, int from, int to, float weight, int write_weight)
{
    if (!net || !net->pool) return -1;
    if (from < 0 || to < 0 || from >= net->pool->max_neurons || to >= net->pool->max_neurons || from == to) return -1;

    neuron_t *nf = nnpool_get_neuron(net->pool, from);
    neuron_t *nt = nnpool_get_neuron(net->pool, to);
    if (!nf || !nt || !nf->active || !nt->active) return -1;

    int md = net->pool->max_degree;
    int fwd_slot = nnpool_find_edge_slot(net->pool, from, to);

    if (fwd_slot >= 0) {
        /* Connection already exists — refresh metadata */
        int eidx = from * md + fwd_slot;
        net->pool->edges[eidx].weight = 0.0f;
        net->pool->edges[eidx].age = 0;

        if (write_weight) {
            int rslot = nnpool_find_edge_slot(net->pool, to, from);
            if (rslot >= 0) {
                float *to_w = nnpool_get_params(net->pool, to);
                if (to_w) to_w[rslot + 1] = weight;
            }
        }
        return eidx;
    }

    if (net->pool->degrees[from] >= md) return -1;

    edge_t *row = nnpool_edge_row(net->pool, from);
    int slot = find_free_edge_slot(row, md);
    if (slot < 0) return -1;

    int eidx = from * md + slot;
    net->pool->degrees[from]++;

    edge_t *e = &net->pool->edges[eidx];
    e->from = from;
    e->to = to;
    e->weight = 0.0f;
    e->age = 0;
    e->active = 1;

    /* Always try to create reverse for easy neighbor walking */
    int rslot = nnpool_find_edge_slot(net->pool, to, from);
    if (rslot < 0 && net->pool->degrees[to] < md) {
        edge_t *rrow = nnpool_edge_row(net->pool, to);
        rslot = find_free_edge_slot(rrow, md);
        if (rslot >= 0) {
            net->pool->degrees[to]++;
            int re = to * md + rslot;
            edge_t *re_e = &net->pool->edges[re];
            re_e->from = to;
            re_e->to = from;
            re_e->weight = 0.0f;
            re_e->age = 0;
            re_e->active = 1;

            if (write_weight) {
                float *to_w = nnpool_get_params(net->pool, to);
                if (to_w) to_w[rslot + 1] = weight;
            }
        }
    } else if (rslot >= 0 && write_weight) {
        float *to_w = nnpool_get_params(net->pool, to);
        if (to_w) to_w[rslot + 1] = weight;
    }

    return eidx;
}

static void remove_bidirectional_edge(network_t *net, int from, int to, int clear_weight)
{
    if (!net || !net->pool) return;
    int md = net->pool->max_degree;

    int slot = nnpool_find_edge_slot(net->pool, from, to);
    if (slot >= 0) {
        edge_t *row = nnpool_edge_row(net->pool, from);
        row[slot].active = 0;
        row[slot].to = -1;
        net->pool->degrees[from]--;
        net->pool->edges[from * md + slot].active = 0;
    }

    int rslot = nnpool_find_edge_slot(net->pool, to, from);
    if (rslot >= 0) {
        edge_t *rrow = nnpool_edge_row(net->pool, to);
        rrow[rslot].active = 0;
        rrow[rslot].to = -1;
        net->pool->degrees[to]--;
        net->pool->edges[to * md + rslot].active = 0;

        if (clear_weight) {
            clear_weight_slot(net->pool, to, rslot);
        }
    }
}

network_t* network_create(int max_neurons, int input_dim, int max_degree)
{
    if (max_neurons <= 0) return NULL;
    if (input_dim <= 0) return NULL;
    if (max_degree <= 0) max_degree = SONN_DEFAULT_MAX_DEGREE;  /* from nnpool.h */

    network_t *net = (network_t*)calloc(1, sizeof(network_t));
    if (!net) return NULL;

    net->pool = nnpool_create(max_neurons, input_dim, max_degree);
    if (!net->pool) {
        free(net);
        return NULL;
    }

    net->current_neurons = 0;
    net->max_neurons = max_neurons;
    net->max_degree = max_degree;
    net->input_dim = input_dim;
    return net;
}

void network_destroy(network_t *net)
{
    if (!net) return;

    if (net->pool) {
        nnpool_destroy(net->pool);
    }
    free(net);
}

int network_add_neuron(network_t *net, activaton_t type)
{
    if (!net || !net->pool) return -1;

    /* All neuron semantic operations are performed in the Network layer.
     * Network is only responsible for acquiring storage slots and organizing the network.
     * Neuron IDs are pre-generated by the Pool at creation time for all slots.
     * Network has no authority to assign or modify IDs.
     */
    int slot = nnpool_acquire_slot(net->pool);
    if (slot < 0) return -1;

    neuron_t *n = nnpool_get_neuron(net->pool, slot);
    float *params = nnpool_get_params(net->pool, slot);

    edge_t *erow = nnpool_edge_row(net->pool, slot);
    if (erow) {
        for (int s = 0; s < net->pool->max_degree; s++) {
            erow[s].active = 0;
            erow[s].to = -1;
        }
    }
    net->pool->degrees[slot] = 0;

    /* Activate the neuron (ID was already assigned at Pool creation time) */
    if (n) {
        neuron_activate(n, type, params, net->pool->input_dim);
    }

    net->current_neurons++;
    return slot;
}

void network_remove_neuron(network_t *net, int id)
{
    if (!net || !net->pool) return;
    if (id < 0 || id >= net->pool->max_neurons) return;

    neuron_t *n = nnpool_get_neuron(net->pool, id);
    if (!n || !n->active) return;

    neuron_deactivate(n);

    /* Remove all incident edges (semantic cleanup).
     * Use remove_edge so weight clearing (if any) is handled consistently.
     */
    int md = net->pool->max_degree;
    edge_t *erow = nnpool_edge_row(net->pool, id);
    if (erow) {
        for (int s = 0; s < md; s++) {
            if (erow[s].active) {
                int to = erow[s].to;
                network_remove_edge(net, id, to);
            }
        }
    }

    /* Clean incoming edges from other neurons (reverse direction) */
    for (int i = 0; i < net->pool->max_neurons; i++) {
        if (i == id) continue;
        neuron_t *other = nnpool_get_neuron(net->pool, i);
        if (!other || !other->active) continue;
        edge_t *orow = nnpool_edge_row(net->pool, i);
        if (!orow) continue;
        for (int s = 0; s < md; s++) {
            if (orow[s].active && orow[s].to == id) {
                network_remove_edge(net, i, id);
            }
        }
    }

    /* Release the raw slot back to the pool (pure memory operation) */
    nnpool_release_slot(net->pool, id);
    net->current_neurons--;
}

int network_add_edge(network_t *net, int from, int to, float weight)
{
    /* Weighted connection path (mainly for layered FFNN).
     * Writes into receiver's weights slot per Option A.
     */
    return add_bidirectional_edge(net, from, to, weight, 1);
}

int network_add_edge_topology(network_t *net, int from, int to)
{
    /* Pure topology path for self-organizing networks (GNG etc).
     * Does not touch the weights array.
     */
    return add_bidirectional_edge(net, from, to, 0.0f, 0);
}

void network_remove_edge(network_t *net, int from, int to)
{
    /* Weighted remove path — clears weight slot (for FFNN) */
    remove_bidirectional_edge(net, from, to, 1);
}

int network_get_neighbors(network_t *net, int id, int *out, int max_out)
{
    if (!net || !net->pool || id < 0 || !out || max_out <= 0) return 0;
    neuron_t *n = nnpool_get_neuron(net->pool, id);
    if (!n || !n->active) return 0;

    int md = net->pool->max_degree;
    edge_t *erow = nnpool_edge_row(net->pool, id);
    if (!erow) return 0;

    int count = 0;
    for (int s = 0; s < md && count < max_out; s++) {
        if (erow[s].active) out[count++] = erow[s].to;
    }
    return count;
}

uint64_t network_get_neuronid(network_t *net, int slot)
{
    if (!net || !net->pool) return 0;
    neuron_t *n = nnpool_get_neuron(net->pool, slot);
    return n ? n->id : 0;
}

float network_prototype_distance(network_t *net, int neuron_id, const float *input)
{
    if (!net || !net->pool || !input || neuron_id < 0) return -1.0f;

    neuron_t *n = nnpool_get_neuron(net->pool, neuron_id);
    if (!n || !n->active || !n->weights) return -1.0f;

    float dist = 0.0f;
    int dim = n->input_dim;
    for (int i = 0; i < dim; i++) {
        float d = input[i] - n->weights[i];
        dist += d * d;
    }
    return dist;   /* squared Euclidean (common in GNG for speed) */
}

int network_find_bmu(network_t *net, const float *input)
{
    return network_find_bmu2(net, input, NULL);
}

int network_find_bmu2(network_t *net, const float *input, int *second_bmu)
{
    if (!net || !net->pool || !input) return -1;

    int best = -1;
    int second = -1;
    float best_dist = 1e30f;
    float second_dist = 1e30f;
    int dim = net->pool->input_dim;

    for (int i = 0; i < net->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(net->pool, i);
        if (!n || !n->active || !n->weights) continue;

        /* Inline squared Euclidean (prototype only) */
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

void network_accumulate_error(network_t *net, int neuron_id, float err)
{
    if (!net || !net->pool || neuron_id < 0) return;

    neuron_t *n = nnpool_get_neuron(net->pool, neuron_id);
    if (n && n->active) {
        n->error += err;
    }
}

void network_adapt_prototype(network_t *net, int neuron_id, const float *input, float epsilon)
{
    if (!net || !net->pool || !input || neuron_id < 0 || epsilon <= 0) return;

    neuron_t *n = nnpool_get_neuron(net->pool, neuron_id);
    if (!n || !n->active || !n->weights) return;

    int dim = n->input_dim;
    for (int i = 0; i < dim; i++) {
        n->weights[i] += epsilon * (input[i] - n->weights[i]);
    }
}

void network_age_edges(network_t *net, int neuron_id)
{
    if (!net || !net->pool) return;

    int md = net->pool->max_degree;

    if (neuron_id >= 0) {
        /* Age only edges of this neuron */
        edge_t *row = nnpool_edge_row(net->pool, neuron_id);
        if (!row) return;
        for (int s = 0; s < md; s++) {
            if (row[s].active) {
                row[s].age += 1.0f;
            }
        }
    } else {
        /* Age all edges in the network */
        for (int i = 0; i < net->pool->max_neurons; i++) {
            edge_t *row = nnpool_edge_row(net->pool, i);
            if (!row) continue;
            for (int s = 0; s < md; s++) {
                if (row[s].active) {
                    row[s].age += 1.0f;
                }
            }
        }
    }
}

void network_reset_edge_age(network_t *net, int from, int to)
{
    if (!net || !net->pool) return;

    int slot = nnpool_find_edge_slot(net->pool, from, to);
    if (slot >= 0) {
        int md = net->pool->max_degree;
        net->pool->edges[from * md + slot].age = 0.0f;
    }

    /* Also reset reverse if exists */
    int rslot = nnpool_find_edge_slot(net->pool, to, from);
    if (rslot >= 0) {
        int md = net->pool->max_degree;
        net->pool->edges[to * md + rslot].age = 0.0f;
    }
}

int network_remove_old_edges(network_t *net, float max_age)
{
    if (!net || !net->pool || max_age <= 0) return 0;

    int removed = 0;
    int md = net->pool->max_degree;

    for (int i = 0; i < net->pool->max_neurons; i++) {
        edge_t *row = nnpool_edge_row(net->pool, i);
        if (!row) continue;

        for (int s = 0; s < md; s++) {
            if (row[s].active && row[s].age > max_age) {
                int to = row[s].to;
                if (to > i) {   /* count & remove each undirected connection only once */
                    remove_bidirectional_edge(net, i, to, 1);
                    removed++;
                } else {
                    /* still remove the other direction if we see it first */
                    remove_bidirectional_edge(net, i, to, 1);
                }
            }
        }
    }
    return removed;
}

float network_get_error(network_t *net, int neuron_id)
{
    if (!net || !net->pool || neuron_id < 0) return 0.0f;
    neuron_t *n = nnpool_get_neuron(net->pool, neuron_id);
    if (n && n->active) return n->error;
    return 0.0f;
}

int network_find_highest_error(network_t *net)
{
    if (!net || !net->pool) return -1;

    int best = -1;
    float best_err = -1.0f;

    for (int i = 0; i < net->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(net->pool, i);
        if (n && n->active && n->error > best_err) {
            best_err = n->error;
            best = i;
        }
    }
    return best;
}

float network_get_edge_age(network_t *net, int from, int to)
{
    if (!net || !net->pool) return -1.0f;
    int slot = nnpool_find_edge_slot(net->pool, from, to);
    if (slot < 0) return -1.0f;
    int md = net->pool->max_degree;
    return net->pool->edges[from * md + slot].age;
}

void network_decay_errors(network_t *net, float factor)
{
    if (!net || !net->pool) return;
    if (factor < 0) factor = 0;

    for (int i = 0; i < net->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(net->pool, i);
        if (n && n->active) {
            n->error *= factor;
        }
    }
}
void network_adapt_bmu_and_neighbors(network_t *net, int bmu,
                                      const float *input,
                                      float epsilon_bmu, float epsilon_n)
{
    if (!net || !input || bmu < 0 || epsilon_bmu <= 0) return;

    /* Adapt BMU */
    network_adapt_prototype(net, bmu, input, epsilon_bmu);

    /* Adapt neighbors — use actual max_degree to avoid hardcoding */
    int md = (net->pool && net->pool->max_degree > 0) ? net->pool->max_degree : 64;
    if (md > 256) md = 256;   /* safety cap for stack buffer */
    int buf[256];
    int ncount = network_get_neighbors(net, bmu, buf, md);
    for (int i = 0; i < ncount; i++) {
        int nb = buf[i];
        if (nb >= 0 && epsilon_n > 0) {
            network_adapt_prototype(net, nb, input, epsilon_n);
        }
    }
}
int network_insert_between(network_t *net, int a, int b, activaton_t type)
{
    if (!net || !net->pool || a < 0 || b < 0) return -1;

    neuron_t *na = nnpool_get_neuron(net->pool, a);
    neuron_t *nb = nnpool_get_neuron(net->pool, b);
    if (!na || !nb || !na->active || !nb->active) return -1;

    int new_id = network_add_neuron(net, type);
    if (new_id < 0) return -1;

    neuron_t *nn = nnpool_get_neuron(net->pool, new_id);
    if (!nn || !nn->weights) {
        network_remove_neuron(net, new_id);
        return -1;
    }

    /* Set prototype to average of a and b */
    int dim = nn->input_dim;
    for (int i = 0; i < dim; i++) {
        float va = (na->weights && i < na->input_dim) ? na->weights[i] : 0.0f;
        float vb = (nb->weights && i < nb->input_dim) ? nb->weights[i] : 0.0f;
        nn->weights[i] = 0.5f * (va + vb);
    }

    /* Remove old direct connection between a and b (typical GNG step) */
    network_remove_edge(net, a, b);

    /* Add topology connections (no weight pollution) */
    network_add_edge_topology(net, a, new_id);
    network_add_edge_topology(net, b, new_id);

    return new_id;
}
