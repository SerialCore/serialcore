/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_GNG
#define SERIALCORE_SONN_GNG

#include <serialcore/sonn/sonn.h>

/* GNG — Growing Neural Gas (Fritzke, 1995) on the generic SONN graph.
 * Error, edge age, and per-unit readout live in this object, not on neuron_t. */

/* Default GNG policy knobs. */
#define GNG_DEFAULT_INSERT_INTERVAL 50
#define GNG_DEFAULT_MAX_AGE         50.0f
#define GNG_DEFAULT_ERROR_DECAY     0.99f

/* Squared distance is the default. prototype and x each have `dim` floats. */
typedef float (*gng_distance_fn)(const float *prototype, const float *x, int dim, void *userdata);

typedef struct gng {
    sonn_t *net;              /* borrowed; caller always owns it, including after gng_load */
    int     insert_interval;  /* insert every N observations; 0 disables insertion */
    float   max_age;          /* edge age beyond which it is pruned */
    float   error_decay;      /* per-step multiplier applied to every unit error */
    int     observe_count;

    float  *error;            /* [max_neurons], indexed by neuron id */
    float  *edge_age;         /* [max_neurons * max_degree], same slot order as pool edges */
    float  *readout;          /* [max_neurons * output_dim], local supervised label */

    gng_distance_fn distance;
    void           *distance_userdata;
} gng_t;

/* Lifecycle. g borrows net. gng_destroy does not free it. */
gng_t *gng_create(sonn_t *net, int insert_interval, float max_age, float error_decay);
void gng_destroy(gng_t *g);

/* `insert_interval` < 0 restores the default; 0 turns insertion off.
 * `max_age` or `error_decay` <= 0 restores that knob's default. */
void gng_configure(gng_t *g, int insert_interval, float max_age, float error_decay);

/* NULL fn restores squared Euclidean distance. */
void gng_set_distance(gng_t *g, gng_distance_fn fn, void *userdata);

/* Squared distance (or the hooked metric) between input and a unit prototype. */
float gng_prototype_distance(gng_t *g, int neuron_id, const float *input);

/* Best matching interior unit, or -1 if none exist.
 * gng_find_bmu2 also writes the second-best id (*second_bmu = -1 if absent). */
int gng_find_bmu(gng_t *g, const float *input);
int gng_find_bmu2(gng_t *g, const float *input, int *second_bmu);

/* Move a prototype toward `input`. Geometry only; no GNG state. */
void gng_adapt_prototype(sonn_t *s, int neuron_id, const float *input, float epsilon);

/* Adapt the BMU by epsilon_bmu and its interior neighbors by epsilon_n. */
void gng_adapt_bmu_and_neighbors(sonn_t *s, int bmu, const float *input, float epsilon_bmu, float epsilon_n);

void gng_accumulate_error(gng_t *g, int neuron_id, float err);
float gng_get_error(gng_t *g, int neuron_id);
int gng_find_highest_error(gng_t *g);
void gng_decay_errors(gng_t *g, float factor);

/* neuron_id >= 0 ages that unit's edges (both stored directions, one logical step).
 * neuron_id < 0 ages every undirected edge once. */
void gng_age_edges(gng_t *g, int neuron_id);
void gng_reset_edge_age(gng_t *g, int from, int to);
float gng_get_edge_age(gng_t *g, int from, int to);

/* Drop edges whose age exceeds max_age. Returns how many undirected edges were removed. */
int gng_remove_old_edges(gng_t *g, float max_age);

/* Insert a unit halfway between a and b. Rolls back if either new edge cannot be added.
 * Returns the new id, or -1. */
int gng_insert_between(gng_t *g, int a, int b, activaton_t type);

/* One Fritzke step. `y` may be NULL. When `y` is set, the BMU's readout moves toward it.
 * The first two observations seed two connected interior units. */
int gng_observe(gng_t *g, const float *x, const float *y, float eps_bmu, float eps_n, float eps_out);

/* Copy the BMU readout into `out` (output_dim floats). Returns -1 if no interior unit exists. */
int gng_predict(gng_t *g, const float *x, float *out);

/* Graph plus GNG state. gng_load allocates the sonn and writes it to *net_out.
 * Caller owns both: sonn_destroy(*net_out) and gng_destroy(g). gng borrows the sonn. */
int gng_save(const gng_t *g, const char *json_path, const char *bin_path);
gng_t *gng_load(const char *json_path, const char *bin_path, sonn_t **net_out);

#endif
