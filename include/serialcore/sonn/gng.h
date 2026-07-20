/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_GNG
#define SERIALCORE_SONN_GNG

#include <serialcore/sonn/sonn.h>

/*
 * GNG — Growing Neural Gas (Fritzke, 1995) algorithm layer built on top of
 * the generic SONN plumbing in <serialcore/sonn/sonn.h>.
 *
 * This translation unit is the first of potentially several self-organizing
 * algorithm layers (GNG, SOM, NG, ESS, ...) that share the same underlying
 * network. Each algorithm owns its own driver function; the SONN core never
 * calls any algorithm code.
 *
 * A `gng_t` wraps a caller-owned `sonn_t *net` together with the GNG
 * policy knobs that the generic SONN no longer knows about:
 *   - insert_interval : insert a new interior neuron every N observations.
 *   - max_age          : edges older than this are pruned each observation.
 *   - error_decay      : error scaling applied each observation (e.g. 0.99).
 *
 * Lifecycle: create a sonn_t with sonn_create(), then create a gng_t on top
 * of it; destroy the gng_t first, then destroy the sonn_t. The gng does not
 * own the sonn.
 */

typedef struct gng {
    sonn_t *net;              /* underlying generic SO network (owned by caller) */
    int     insert_interval;  /* insert a new interior neuron every N obs (0 = off) */
    float   max_age;          /* edge age beyond which it is pruned */
    float   error_decay;      /* per-step error decay factor */
    int     observe_count;    /* observations since the last insertion */
} gng_t;

/* Default GNG policy knobs. */
#define GNG_DEFAULT_INSERT_INTERVAL 50
#define GNG_DEFAULT_MAX_AGE         50.0f
#define GNG_DEFAULT_ERROR_DECAY     0.99f

/* Lifecycle. The gng borrows the sonn pointer; caller still owns it. */
gng_t *gng_create(sonn_t *net, int insert_interval, float max_age, float error_decay);
void gng_destroy(gng_t *g);

/* Build a gng with the default policy knobs. */
gng_t *gng_create_default(sonn_t *net);

/* Update the policy knobs at runtime. Passing 0/0/0 restores the defaults. */
void gng_configure(gng_t *g, int insert_interval, float max_age, float error_decay);

/* Compute squared Euclidean distance between input and neuron's prototype. */
float gng_prototype_distance(sonn_t *s, int neuron_id, const float *input);

/* Find the Best Matching Unit (interior neuron closest to input). Returns the
 * neuron id, or -1 if no interior neurons exist. */
int gng_find_bmu(sonn_t *s, const float *input);

/* Find BMU and second BMU (useful for GNG insertion). Writes second BMU to
 * *second_bmu if provided. Returns BMU id, or -1 if no interior neurons
 * exist. */
int gng_find_bmu2(sonn_t *s, const float *input, int *second_bmu);

/* Move neuron's prototype (weights) toward the input vector. */
void gng_adapt_prototype(sonn_t *s, int neuron_id, const float *input, float epsilon);

/* Adapt the BMU toward input with epsilon_bmu, and all its direct topological
 * neighbors with epsilon_n. This is the classic "BMU + neighborhood" update
 * step in many SO algorithms. */
void gng_adapt_bmu_and_neighbors(sonn_t *s, int bmu, const float *input, float epsilon_bmu, float epsilon_n);

/* Add error to a neuron (for GNG error accumulation). */
void gng_accumulate_error(sonn_t *s, int neuron_id, float err);

/* Query the accumulated error of a neuron. */
float gng_get_error(sonn_t *s, int neuron_id);

/* Find the active interior neuron with the highest accumulated error. */
int gng_find_highest_error(sonn_t *s);

/* Multiply every neuron's accumulated error by factor (typically < 1.0).
 * Call periodically to prevent unbounded error growth. */
void gng_decay_errors(sonn_t *s, float factor);

/* Increment age on all edges of a neuron (or globally if id == -1). */
void gng_age_edges(sonn_t *s, int neuron_id);

/* Reset the age of the edge from -> to to 0. */
void gng_reset_edge_age(sonn_t *s, int from, int to);

/* Get the current age of the edge from -> to (returns -1 if no edge). */
float gng_get_edge_age(sonn_t *s, int from, int to);

/* Remove edges whose age exceeds max_age. Returns number of edges removed. */
int gng_remove_old_edges(sonn_t *s, float max_age);

/* Insert a new interior neuron with prototype = average of a and b.
 * Removes the direct edge between a and b (if any).
 * Connects the new neuron topologically to both a and b.
 * Returns the id of the new neuron, or -1 on failure. */
int gng_insert_between(sonn_t *s, int a, int b, activaton_t type);

/* Present one sample to the network and let it grow / adapt */
int gng_observe(gng_t *g, const float *x, const float *y, float eps_bmu, float eps_n, float eps_out);

#endif