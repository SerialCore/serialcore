/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit test: SONN auto-growth on the 2-D XOR input space, then save as
 * meta JSON + binary params, reload, and re-run BMU queries.
 */

#include <serialcore/sonn/sonn.h>
#include <serialcore/sonn/gng.h>
#include <serialcore/math/xoshiross.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const float X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
static const char *META_PATH = "build/test_gng_xor_model.json";
static const char *BIN_PATH  = "build/test_gng_xor_model.bin";

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

static void snapshot(sonn_t *s, int *interior, float *total_err,
                     int *edges, float *mean_dist)
{
    int   inter = s->current_neurons - s->in_count - s->out_count;
    float err   = 0.0f;
    int   ed    = 0;
    float dist  = 0.0f;

    int md = s->pool->max_degree;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (!n || !n->active) continue;

        if (!sonn_is_interior(s, i)) continue;
        err += n->error;

        edge_t *row = nnpool_edge_row(s->pool, i);
        for (int j = 0; j < md; j++) {
            if (row[j].active && row[j].to > i) ed++;
        }
    }

    for (int i = 0; i < 4; i++) {
        int bmu = gng_find_bmu(s, X[i]);
        if (bmu >= 0) dist += gng_prototype_distance(s, bmu, X[i]);
    }

    if (interior)  *interior = inter;
    if (total_err) *total_err = err;
    if (edges)     *edges = ed;
    if (mean_dist) *mean_dist = dist / 4.0f;
}

int main(void)
{
    xoshiro_seed(0xC0FFEEULL);

    const int input_dim  = 2;
    const int output_dim = 1;
    const int max_neurons = 256;
    const int max_degree = 64;

    sonn_t *s = sonn_create(input_dim, output_dim, max_neurons, max_degree, GELU);
    int failed = 0;
    failed += check(s != NULL, "sonn_create succeeded");

    int in_start = -1, in_count = -1, out_start = -1, out_count = -1;
    sonn_get_input_range(s, &in_start, &in_count);
    sonn_get_output_range(s, &out_start, &out_count);
    failed += check(in_start == 0 && in_count == 2, "input anchors = 2 at slots 0,1");
    failed += check(out_start == 2 && out_count == 1, "output anchor = 1 at slot 2");

    int interior_before = s->current_neurons - s->in_count - s->out_count;
    failed += check(interior_before == 0, "no interior neurons at start");

    gng_t *gng = gng_create(s, /*insert_interval=*/8,
                            /*max_age=*/30.0f,
                            /*error_decay=*/0.95f);
    failed += check(gng != NULL, "gng_create succeeded");

    const int epochs        = 40;
    const float eps_bmu     = 0.15f;
    const float eps_n       = 0.05f;

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 4; i++) {
            gng_observe(gng, X[i], NULL, eps_bmu, eps_n, 0.0f);
        }
        if ((e % 10) == 0 || e == epochs - 1) {
            int inter; float terr; int ed; float mdist;
            snapshot(s, &inter, &terr, &ed, &mdist);
            printf("epoch %4d  interior=%-3d total=%-3d edges=%-3d "
                   "mean_bmu_dist=%.6f  total_err=%.6f\n",
                   e, inter, s->current_neurons, ed, mdist, terr);
        }
    }

    int interior_after = s->current_neurons - s->in_count - s->out_count;
    int total_after    = s->current_neurons;
    int edges_after = 0;
    float mean_dist_after = 0.0f;
    snapshot(s, NULL, NULL, &edges_after, &mean_dist_after);

    failed += check(interior_after > 0,
                    "interior neurons were grown by gng_observe()");
    failed += check(total_after == in_count + out_count + interior_after,
                    "total = input + output + interior");

    int saved_bmu[4];
    float saved_dist[4];
    printf("  final per-pattern coverage:\n");
    for (int i = 0; i < 4; i++) {
        saved_bmu[i] = gng_find_bmu(s, X[i]);
        saved_dist[i] = gng_prototype_distance(s, saved_bmu[i], X[i]);
        printf("    X=(%g,%g) -> bmu=%-3d dist=%.6f\n",
               X[i][0], X[i][1], saved_bmu[i], saved_dist[i]);
    }

    int bmu = gng_find_bmu(s, X[0]);
    failed += check(bmu >= 0, "gng_find_bmu returns valid id");
    int is_in  = (bmu >= in_start  && bmu < in_start  + in_count);
    int is_out = (bmu >= out_start && bmu < out_start + out_count);
    failed += check(!is_in && !is_out, "BMU is an interior neuron");

    /* Save meta JSON + binary params, destroy, reload, re-run BMU. */
    failed += check(sonn_save(s, META_PATH, BIN_PATH) == 0, "sonn_save");
    gng_destroy(gng);
    sonn_destroy(s);
    s = NULL;
    gng = NULL;

    sonn_t *loaded = sonn_load(META_PATH, BIN_PATH);
    failed += check(loaded != NULL, "sonn_load");
    if (loaded) {
        int lin = 0, ledges = 0;
        float ldist = 0.0f;
        snapshot(loaded, &lin, NULL, &ledges, &ldist);

        failed += check(loaded->input_dim == input_dim && loaded->output_dim == output_dim,
                        "loaded dims match");
        failed += check(lin == interior_after, "loaded interior count matches");
        failed += check(loaded->current_neurons == total_after, "loaded total neurons match");
        failed += check(ledges == edges_after, "loaded edge count matches");
        failed += check(fabsf(ldist - mean_dist_after) < 1e-5f, "loaded mean BMU dist matches");

        printf("  reloaded per-pattern coverage:\n");
        for (int i = 0; i < 4; i++) {
            int lbmu = gng_find_bmu(loaded, X[i]);
            float ld = gng_prototype_distance(loaded, lbmu, X[i]);
            char buf[96];
            printf("    X=(%g,%g) -> bmu=%-3d dist=%.6f\n",
                   X[i][0], X[i][1], lbmu, ld);
            snprintf(buf, sizeof(buf), "loaded BMU for X[%d] matches", i);
            failed += check(lbmu == saved_bmu[i], buf);
            snprintf(buf, sizeof(buf), "loaded dist for X[%d] matches", i);
            failed += check(fabsf(ld - saved_dist[i]) < 1e-5f, buf);
        }

        sonn_destroy(loaded);
    }

    if (failed) {
        printf("test_gng_xor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_gng_xor: PASS\n\n");
    return 0;
}
