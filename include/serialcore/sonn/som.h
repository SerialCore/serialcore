/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_SOM
#define SERIALCORE_SONN_SOM

#include <serialcore/sonn/sonn.h>

/* SOM — Kohonen self-organizing map on a fixed rectangular grid.
 * Grid coordinates and the annealing schedule live here, not on neuron_t.
 * Neighborhood is grid distance. The SONN edge list is not used. */

#define SOM_DEFAULT_N_STEPS    1000
#define SOM_DEFAULT_EPS_START  0.30f
#define SOM_DEFAULT_EPS_END    0.01f
#define SOM_DEFAULT_SIGMA_END  0.50f

/* Squared distance is the default. prototype and x each have `dim` floats. */
typedef float (*som_distance_fn)(const float *prototype, const float *x, int dim, void *userdata);

typedef struct som {
    sonn_t *net;             /* borrowed; caller always owns it, including after som_load */
    int     rows;
    int     cols;
    int     n_units;         /* rows * cols */
    int     n_steps;         /* decay horizon; step 0 uses the start values, step n_steps-1 the end */
    int     observe_count;

    float   eps_start;
    float   eps_end;
    float   sigma_start;     /* initial Gaussian radius, in grid steps */
    float   sigma_end;

    int    *unit_ids;        /* [n_units], row-major neuron ids */
    int    *row_of;          /* [max_neurons], -1 when the id is not on this map */
    int    *col_of;

    som_distance_fn distance;
    void           *distance_userdata;
} som_t;

/* Place a rows*cols grid of interior neurons. sigma_start <= 0 uses half the longer side.
 * Any other value <= 0 selects that knob's default. m borrows net; som_destroy does not free it. */
som_t *som_create(sonn_t *net, int rows, int cols, int n_steps,
                  float eps_start, float eps_end, float sigma_start, float sigma_end);
void som_destroy(som_t *m);

void som_configure(som_t *m, int n_steps, float eps_start, float eps_end,
                   float sigma_start, float sigma_end);

/* NULL fn restores squared Euclidean distance. */
void som_set_distance(som_t *m, som_distance_fn fn, void *userdata);

/* Schedule values at the current observe_count. */
float som_epsilon(const som_t *m);
float som_sigma(const som_t *m);

/* Neuron id at a grid cell, or -1. */
int som_unit_at(const som_t *m, int row, int col);

/* Squared distance (or the hooked metric) between input and one map unit. */
float som_prototype_distance(som_t *m, int neuron_id, const float *input);

/* Best matching map unit, or -1. */
int som_find_bmu(som_t *m, const float *input);

/* One online Kohonen step. w_i += ε(t) * exp(-||r_i - r_c||^2 / (2 σ(t)^2)) * (x - w_i). */
int som_observe(som_t *m, const float *x);

/* Graph plus the grid and schedule. som_load allocates the sonn and writes it to *net_out.
 * Caller owns both: sonn_destroy(*net_out) and som_destroy(m). */
int som_save(const som_t *m, const char *json_path, const char *bin_path);
som_t *som_load(const char *json_path, const char *bin_path, sonn_t **net_out);

#endif
