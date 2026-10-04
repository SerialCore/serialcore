/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_NGAS
#define SERIALCORE_SONN_NGAS

#include <serialcore/sonn/sonn.h>

/* Neural Gas — Martinetz & Schulten, 1991.
 * A fixed set of prototypes. Each step ranks them by distance to the sample
 * and adapts every unit by exp(-rank / λ(t)). No edges are created.
 * Competitive Hebbian topology is a separate algorithm and is not applied here. */

#define NGAS_DEFAULT_N_STEPS      1000
#define NGAS_DEFAULT_EPS_START    0.30f
#define NGAS_DEFAULT_EPS_END      0.01f
#define NGAS_DEFAULT_LAMBDA_END   0.01f

/* Squared distance is the default. prototype and x each have `dim` floats. */
typedef float (*ngas_distance_fn)(const float *prototype, const float *x, int dim, void *userdata);

typedef struct ngas {
    sonn_t *net;             /* borrowed; caller always owns it, including after ngas_load */
    int     n_units;
    int     n_steps;
    int     observe_count;

    float   eps_start;
    float   eps_end;
    float   lambda_start;    /* rank scale; <= 0 at create time means n_units */
    float   lambda_end;

    int    *unit_ids;        /* [n_units] neuron ids */

    ngas_distance_fn distance;
    void            *distance_userdata;
} ngas_t;

/* Add n_units interior neurons. A value <= 0 selects that knob's default.
 * lambda_start <= 0 selects (float)n_units. m borrows net; ngas_destroy does not free it. */
ngas_t *ngas_create(sonn_t *net, int n_units, int n_steps,
                    float eps_start, float eps_end, float lambda_start, float lambda_end);
void ngas_destroy(ngas_t *m);

void ngas_configure(ngas_t *m, int n_steps, float eps_start, float eps_end,
                    float lambda_start, float lambda_end);

/* NULL fn restores squared Euclidean distance. */
void ngas_set_distance(ngas_t *m, ngas_distance_fn fn, void *userdata);

/* Schedule values at the current observe_count. */
float ngas_epsilon(const ngas_t *m);
float ngas_lambda(const ngas_t *m);

/* Neuron id of unit index, or -1. */
int ngas_unit_at(const ngas_t *m, int index);

float ngas_prototype_distance(ngas_t *m, int neuron_id, const float *input);

/* Best matching gas unit (rank 0), or -1. */
int ngas_find_bmu(ngas_t *m, const float *input);

/* One Neural Gas step. Rank k_i is the number of units strictly closer than i.
 * w_i += ε(t) * exp(-k_i / λ(t)) * (x - w_i). */
int ngas_observe(ngas_t *m, const float *x);

/* Graph plus the unit list and schedule. ngas_load allocates the sonn and writes it to *net_out.
 * Caller owns both: sonn_destroy(*net_out) and ngas_destroy(m). */
int ngas_save(const ngas_t *m, const char *json_path, const char *bin_path);
ngas_t *ngas_load(const char *json_path, const char *bin_path, sonn_t **net_out);

#endif
