/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_SONN
#define SERIALCORE_SONN_SONN

#include <serialcore/cJSON.h>
#include <serialcore/math/activaton.h>
#include <serialcore/sonn/nnpool.h>

#define SONN_DEFAULT_MAX_DEGREE 64

typedef struct sonn {
    nnpool_t    *pool;              /* underlying memory pool (capacity and storage only) */

    int         current_neurons;    /* number of neurons currently in use */
    int         max_neurons;
    int         max_degree;
    int         input_dim;
    int         output_dim;

    int         in_start;
    int         in_count;
    int         out_start;
    int         out_count;
} sonn_t;

/* Lifecycle */
sonn_t* sonn_create(int input_dim, int output_dim, int max_neurons, int max_degree, activaton_t type);
void sonn_destroy(sonn_t *s);

/* Neuron operations */
int sonn_add_neuron(sonn_t *s, activaton_t type);
void sonn_remove_neuron(sonn_t *s, int id);

/* Returns the index into the pool's edge arena for `a`'s half (or -1 on failure). */
int sonn_add_edge(sonn_t *s, int a, int b);
void sonn_remove_edge(sonn_t *s, int a, int b);

/* Interior neurons are the ones algorithm layers grow and adapt. */
int sonn_is_interior(const sonn_t *s, int id);

/* Queries. get_neighbors stops after max_out entries.
 * foreach_neighbor visits every live neighbor; cb != 0 stops the walk.
 * Both return how many neighbors were visited. */
int sonn_get_neighbors(const sonn_t *s, int id, int *out, int max_out);
typedef int (*sonn_neighbor_cb)(int neighbor, void *userdata);
int sonn_foreach_neighbor(const sonn_t *s, int id, sonn_neighbor_cb cb, void *userdata);

/* Range accessors for the fixed input/output neuron anchors. */
int sonn_get_input_range(const sonn_t *s, int *start, int *count);
int sonn_get_output_range(const sonn_t *s, int *start, int *count);

/* Graph JSON, without the parameter binary. Caller cJSON_Delete's the result. */
cJSON *sonn_to_json(const sonn_t *s);
sonn_t *sonn_from_json(const cJSON *root);

/* Persist meta JSON (dims, active neurons, edges) + nnpool params binary.
 * Edge age and other algorithm state are not part of this file. */
int sonn_save(const sonn_t *s, const char *json_path, const char *bin_path);
sonn_t *sonn_load(const char *json_path, const char *bin_path);

/* Refresh neuron.bias and neuron.weights after the parameter block is overwritten. */
void sonn_rebind_params(sonn_t *s);

#endif