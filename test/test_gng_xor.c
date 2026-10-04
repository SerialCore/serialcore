/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * SONN/GNG: grow on the XOR corners, check symmetric edge ages, then
 * round-trip the graph (sonn_load) and the algorithm state (gng_load).
 * A second net checks that readout follows the BMU instead of a global label.
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

static void snapshot(gng_t *g, int *interior, float *total_err,
                     int *edges, float *mean_dist)
{
    sonn_t *s = g->net;
    int   inter = s->current_neurons - s->in_count - s->out_count;
    float err   = 0.0f;
    int   ed    = 0;
    float dist  = 0.0f;
    int md = s->pool->max_degree;

    for (int i = 0; i < s->pool->max_neurons; i++) {
        neuron_t *n = nnpool_get_neuron(s->pool, i);
        if (!n || !n->active || !sonn_is_interior(s, i)) continue;
        err += gng_get_error(g, i);

        edge_t *row = nnpool_edge_row(s->pool, i);
        for (int j = 0; j < md; j++) {
            if (row[j].active && row[j].to > i) ed++;
        }
    }

    for (int i = 0; i < 4; i++) {
        int bmu = gng_find_bmu(g, X[i]);
        if (bmu >= 0) dist += gng_prototype_distance(g, bmu, X[i]);
    }

    if (interior)  *interior = inter;
    if (total_err) *total_err = err;
    if (edges)     *edges = ed;
    if (mean_dist) *mean_dist = dist / 4.0f;
}

static int ages_symmetric(gng_t *g)
{
    sonn_t *s = g->net;
    int md = s->pool->max_degree;

    for (int id = 0; id < s->pool->max_neurons; id++) {
        edge_t *row = nnpool_edge_row(s->pool, id);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            float a, b;
            if (!row[j].active || row[j].to <= id) continue;
            a = gng_get_edge_age(g, id, row[j].to);
            b = gng_get_edge_age(g, row[j].to, id);
            if (a < 0.0f || fabsf(a - b) > 1e-6f) return 0;
        }
    }
    return 1;
}

static int state_matches(gng_t *a, gng_t *b)
{
    sonn_t *sa = a->net;
    sonn_t *sb = b->net;
    int md = sa->pool->max_degree;

    if (a->observe_count != b->observe_count) return 0;
    if (a->insert_interval != b->insert_interval) return 0;
    if (fabsf(a->max_age - b->max_age) > 1e-6f) return 0;

    for (int id = 0; id < sa->pool->max_neurons; id++) {
        neuron_t *na = nnpool_get_neuron(sa->pool, id);
        neuron_t *nb = nnpool_get_neuron(sb->pool, id);
        int aa = na && na->active;
        int bb = nb && nb->active;
        if (aa != bb) return 0;
        if (!aa || !sonn_is_interior(sa, id)) continue;
        if (fabsf(gng_get_error(a, id) - gng_get_error(b, id)) > 1e-4f) return 0;
        for (int k = 0; k < sa->output_dim; k++) {
            float ra = a->readout[(size_t)id * sa->output_dim + k];
            float rb = b->readout[(size_t)id * sb->output_dim + k];
            if (fabsf(ra - rb) > 1e-4f) return 0;
        }
    }

    for (int id = 0; id < sa->pool->max_neurons; id++) {
        edge_t *row = nnpool_edge_row(sa->pool, id);
        if (!row) continue;
        for (int j = 0; j < md; j++) {
            float aa, bb;
            if (!row[j].active || row[j].to <= id) continue;
            aa = gng_get_edge_age(a, id, row[j].to);
            bb = gng_get_edge_age(b, id, row[j].to);
            if (fabsf(aa - bb) > 1e-4f) return 0;
            if (fabsf(gng_get_edge_age(b, row[j].to, id) - bb) > 1e-4f) return 0;
        }
    }
    return 1;
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
            snapshot(gng, &inter, &terr, &ed, &mdist);
            printf("epoch %4d  interior=%-3d total=%-3d edges=%-3d "
                   "mean_bmu_dist=%.6f  total_err=%.6f\n",
                   e, inter, s->current_neurons, ed, mdist, terr);
        }
    }

    int interior_after = s->current_neurons - s->in_count - s->out_count;
    int total_after    = s->current_neurons;
    int edges_after = 0;
    float mean_dist_after = 0.0f;
    snapshot(gng, NULL, NULL, &edges_after, &mean_dist_after);

    failed += check(interior_after >= 2, "at least the seed pair of interior neurons");
    failed += check(total_after == in_count + out_count + interior_after,
                    "total = input + output + interior");
    failed += check(ages_symmetric(gng), "edge ages match in both directions");

    int saved_bmu[4];
    float saved_dist[4];
    printf("  final per-pattern coverage:\n");
    for (int i = 0; i < 4; i++) {
        saved_bmu[i] = gng_find_bmu(gng, X[i]);
        saved_dist[i] = gng_prototype_distance(gng, saved_bmu[i], X[i]);
        printf("    X=(%g,%g) -> bmu=%-3d dist=%.6f\n",
               X[i][0], X[i][1], saved_bmu[i], saved_dist[i]);
    }

    int bmu = gng_find_bmu(gng, X[0]);
    failed += check(bmu >= 0, "gng_find_bmu returns valid id");
    int is_in  = (bmu >= in_start  && bmu < in_start  + in_count);
    int is_out = (bmu >= out_start && bmu < out_start + out_count);
    failed += check(!is_in && !is_out, "BMU is an interior neuron");

    failed += check(gng_save(gng, META_PATH, BIN_PATH) == 0, "gng_save");

    /* Graph-only load ignores the GNG object and still restores prototypes. */
    sonn_t *loaded = sonn_load(META_PATH, BIN_PATH);
    failed += check(loaded != NULL, "sonn_load");
    if (loaded) {
        gng_t *query = gng_create(loaded, 0, 30.0f, 0.95f);
        int lin = 0, ledges = 0;
        float ldist = 0.0f;
        snapshot(query, &lin, NULL, &ledges, &ldist);

        failed += check(loaded->input_dim == input_dim && loaded->output_dim == output_dim,
                        "loaded dims match");
        failed += check(lin == interior_after, "loaded interior count matches");
        failed += check(loaded->current_neurons == total_after, "loaded total neurons match");
        failed += check(ledges == edges_after, "loaded edge count matches");
        failed += check(fabsf(ldist - mean_dist_after) < 1e-5f, "loaded mean BMU dist matches");

        printf("  reloaded per-pattern coverage:\n");
        for (int i = 0; i < 4; i++) {
            int lbmu = gng_find_bmu(query, X[i]);
            float ld = gng_prototype_distance(query, lbmu, X[i]);
            char buf[96];
            printf("    X=(%g,%g) -> bmu=%-3d dist=%.6f\n",
                   X[i][0], X[i][1], lbmu, ld);
            snprintf(buf, sizeof(buf), "loaded BMU for X[%d] matches", i);
            failed += check(lbmu == saved_bmu[i], buf);
            snprintf(buf, sizeof(buf), "loaded dist for X[%d] matches", i);
            failed += check(fabsf(ld - saved_dist[i]) < 1e-5f, buf);
        }
        gng_destroy(query);
        sonn_destroy(loaded);
    }

    sonn_t *net2 = NULL;
    gng_t *g2 = gng_load(META_PATH, BIN_PATH, &net2);
    failed += check(g2 != NULL && net2 != NULL, "gng_load");
    if (g2) {
        failed += check(state_matches(gng, g2), "loaded GNG error, age, readout, counters");
        for (int i = 0; i < 4; i++) {
            failed += check(gng_find_bmu(g2, X[i]) == saved_bmu[i], "gng_load BMU matches");
        }
        gng_destroy(g2);
        sonn_destroy(net2);
    }

    gng_destroy(gng);
    sonn_destroy(s);

    /* insert_interval 0 stays off; max_age 0 restores the default. */
    {
        sonn_t *cs = sonn_create(2, 1, 16, 4, ReLU);
        gng_t *cg = gng_create(cs, 4, 10.0f, 0.5f);
        gng_configure(cg, 0, 0.0f, 0.0f);
        failed += check(cg->insert_interval == 0, "insert_interval 0 disables insertion");
        failed += check(cg->max_age == GNG_DEFAULT_MAX_AGE, "max_age <= 0 restores default");
        failed += check(cg->error_decay == GNG_DEFAULT_ERROR_DECAY, "error_decay <= 0 restores default");
        gng_destroy(cg);
        sonn_destroy(cs);
    }

    /* Local readout: each corner's BMU tracks that corner's label. */
    {
        sonn_t *rs = sonn_create(2, 1, 32, 8, GELU);
        gng_t *rg = gng_create(rs, /*insert_interval=*/0, 1000.0f, 0.99f);
        const float y0 = 0.0f, y1 = 1.0f;
        float p0 = 0.0f, p1 = 0.0f;
        int inter = 0;

        for (int n = 0; n < 40; n++) {
            gng_observe(rg, X[0], &y0, 0.05f, 0.01f, 0.5f);
            gng_observe(rg, X[1], &y1, 0.05f, 0.01f, 0.5f);
        }
        gng_predict(rg, X[0], &p0);
        gng_predict(rg, X[1], &p1);
        inter = rs->current_neurons - rs->in_count - rs->out_count;
        printf("  readout (0,0)->%.3f  (0,1)->%.3f  interior=%d\n", p0, p1, inter);
        failed += check(inter == 2, "insertion off keeps the seed pair");
        failed += check(p0 < 0.25f && p1 > 0.75f, "BMU readout tracks the local label");
        gng_destroy(rg);
        sonn_destroy(rs);
    }

    if (failed) {
        printf("test_gng_xor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_gng_xor: PASS\n\n");
    return 0;
}
