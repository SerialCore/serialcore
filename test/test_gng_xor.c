/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit test: SONN auto-growth on the 2-D XOR input space.
 *
 * The SONN plumbing no longer contains any algorithm — Growing Neural Gas was
 * split out to src/sonn/gng.c (see <serialcore/sonn/gng.h>). Here we exercise
 * what the GNG layer is intrinsically good at: feed it the four 2-D points of
 * the XOR corners, let gng_observe() add interior neurons on its own, and
 * assert that:
 *
 *   1. The input/output anchors are pre-allocated exactly as requested by
 *      sonn_create(input_dim, output_dim, ...).
 *   2. After observing the corners, the network has grown interior neurons
 *      (sonn_interior_neurons() > 0).
 *   3. Bounding any single sample worked: sonn_find_bmu returns a valid id
 *      interior to the network.
 *
 * This is *not* a "did the network learn XOR" test; that belongs to the FFNN
 * runtime. This test is for the SONN's structural dynamics alone.
 */

#include <serialcore/sonn/sonn.h>
#include <serialcore/sonn/gng.h>
#include <serialcore/math/xoshiross.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* The four corners of the 2-D unit square — the standard XOR input set.
 * For pure self-organizing growth we feed them as unlabeled signals.
 */
static const float X[4][2] = {{0,0},{0,1},{1,0},{1,1}};

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

/* Aggregate growth / quantization metrics for a snapshot.
 *   interior  : number of grown interior neurons
 *   total_err : sum of n->error over all interior neurons
 *   edges     : number of active bidirectional edges (counted once)
 *   mean_dist : mean squared Euclidean distance from each XOR corner to its
 *               BMU (quantization error — the SO analog of MSE).
 */
static void snapshot(sonn_t *s, int *interior, float *total_err,
                     int *edges, float *mean_dist)
{
    int   inter = sonn_interior_neurons(s);
    float err   = 0.0f;
    int   ed    = 0;
    float dist  = 0.0f;

    int md = s->pool->max_degree;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (!n || !n->active) continue;

        if (!sonn_is_interior(s, i)) continue;
        err += n->error;

        /* count each undirected active edge once (only when to > i) */
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
    /* sonn_observe() touches the prototype-block RNG only indirectly via the
     * pool init; still, seed deterministically for reproducible growth. */
    xoshiro_seed(0xC0FFEEULL);

    /* Treat XOR as a (2)->(1) shape: 2 input anchors, 1 output anchor.
     * Allow plenty of growth headroom (max_neurons=256). */
    const int input_dim  = 2;
    const int output_dim = 1;
    const int max_neurons = 256;
    const int max_degree = 64;

    sonn_t *s = sonn_create(input_dim, output_dim, max_neurons, max_degree);
    int failed = 0;
    failed += check(s != NULL, "sonn_create succeeded");

    /* 1. Anchor ranges are exactly input_dim / output_dim. */
    int in_start = -1, in_count = -1, out_start = -1, out_count = -1;
    sonn_get_input_range(s, &in_start, &in_count);
    sonn_get_output_range(s, &out_start, &out_count);
    failed += check(in_start == 0 && in_count == 2, "input anchors = 2 at slots 0,1");
    failed += check(out_start == 2 && out_count == 1, "output anchor = 1 at slot 2");

    /* 2. After creation, only the anchors are alive — no interior neurons. */
    int interior_before = sonn_interior_neurons(s);
    failed += check(interior_before == 0, "no interior neurons at start");

    /* 3. Feed the corners to the network and let it grow. Tighten the growth
     *    cadence so the test takes very few observations to demonstrate growth.
     *    Insert a new neuron every 8 observations. The GNG owns the policy
     *    knobs that the generic SONN no longer knows about. */
    gng_t *gng = gng_create(s, /*insert_interval=*/8,
                            /*max_age=*/30.0f,
                            /*error_decay=*/0.95f);
    failed += check(gng != NULL, "gng_create succeeded");

    const int epochs        = 40;     /* = 40*4 = 160 observations */
    const float eps_bmu     = 0.15f;
    const float eps_n       = 0.05f;

    /* Pure self-organization: y == NULL means no supervised pull on the
     * output anchor. The SONN learns the *structure* of the input space,
     * not the XOR mapping (mapping is the FFNN's job). */
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 4; i++) {
            gng_observe(gng, X[i], NULL, eps_bmu, eps_n, 0.0f);
        }
        if ((e % 10) == 0 || e == epochs - 1) {
            int inter; float terr; int ed; float mdist;
            snapshot(s, &inter, &terr, &ed, &mdist);
            printf("epoch %4d  interior=%-3d total=%-3d edges=%-3d "
                   "mean_bmu_dist=%.6f  total_err=%.6f\n",
                   e, inter, sonn_current_neurons(s), ed, mdist, terr);
        }
    }

    int interior_after = sonn_interior_neurons(s);
    int total_after    = sonn_current_neurons(s);

    failed += check(interior_after > 0,
                    "interior neurons were grown by gng_observe()");
    failed += check(total_after == in_count + out_count + interior_after,
                    "total = input + output + interior");

    /* 4. Per-pattern final readout: for each XOR corner list the BMU
     *    (interior neuron id) and squared distance to it. Cluster
     *    coverage is the SONN's notion of "result". */
    printf("  final per-pattern coverage:\n");
    for (int i = 0; i < 4; i++) {
        int bmu = gng_find_bmu(s, X[i]);
        float d = gng_prototype_distance(s, bmu, X[i]);
        printf("    X=(%g,%g) -> bmu=%-3d dist=%.6f\n",
               X[i][0], X[i][1], bmu, d);
    }

    /* 5. BMU selection works end-to-end and returns an interior neuron
     *    (never one of the fixed input/output anchors). */
    int bmu = gng_find_bmu(s, X[0]);
    failed += check(bmu >= 0, "gng_find_bmu returns valid id");
    int is_in  = (bmu >= in_start  && bmu < in_start  + in_count);
    int is_out = (bmu >= out_start && bmu < out_start + out_count);
    failed += check(!is_in && !is_out, "BMU is an interior neuron");

    gng_destroy(gng);
    sonn_destroy(s);

    if (failed) {
        printf("test_gng_xor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_gng_xor: PASS\n\n");
    return 0;
}