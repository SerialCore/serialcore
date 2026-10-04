/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* One Neural Gas step with three prototypes on a line, then a save/load round trip. */

#include <serialcore/sonn/ngas.h>

#include <math.h>
#include <stdio.h>

static const char *META_PATH = "build/test_ngas_model.json";
static const char *BIN_PATH  = "build/test_ngas_model.bin";

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

static int near2(const float *w, float x, float y)
{
    return fabsf(w[0] - x) < 1e-5f && fabsf(w[1] - y) < 1e-5f;
}

static void write_at(sonn_t *s, int id, float x, float y)
{
    neuron_t *n = nnpool_get_neuron(s->pool, id);
    n->weights[0] = x;
    n->weights[1] = y;
}

int main(void)
{
    const float sample[2] = {0.0f, 0.0f};
    float h1, h2;
    int failed = 0;
    int u0, u1, u2;
    int bmu;
    sonn_t *net2 = NULL;
    ngas_t *loaded;
    neuron_t *n;

    sonn_t *s = sonn_create(2, 1, 16, 4, GELU);
    ngas_t *m = ngas_create(s, 3, /*n_steps=*/8,
                            /*eps*/ 0.5f, 0.5f,
                            /*lambda*/ 1.0f, 1.0f);
    failed += check(s != NULL && m != NULL, "ngas_create 3 units");
    if (!m) {
        sonn_destroy(s);
        printf("test_ngas: FAIL (%d assertions broken)\n", failed);
        return 1;
    }

    failed += check(m->lambda_start == 1.0f, "explicit lambda_start is kept");
    failed += check(fabsf(ngas_epsilon(m) - 0.5f) < 1e-6f, "step 0 uses eps_start");
    failed += check(fabsf(ngas_lambda(m) - 1.0f) < 1e-6f, "step 0 uses lambda_start");

    u0 = ngas_unit_at(m, 0);
    u1 = ngas_unit_at(m, 1);
    u2 = ngas_unit_at(m, 2);
    write_at(s, u0, 0.0f, 0.0f);
    write_at(s, u1, 1.0f, 0.0f);
    write_at(s, u2, 3.0f, 0.0f);

    bmu = ngas_find_bmu(m, sample);
    failed += check(bmu == u0, "BMU is the closest prototype");
    failed += check(ngas_observe(m, sample) == 0, "ngas_observe");

    h1 = expf(-1.0f);
    h2 = expf(-2.0f);
    n = nnpool_get_neuron(s->pool, u0);
    failed += check(near2(n->weights, 0.0f, 0.0f), "rank 0 stays on the sample");
    n = nnpool_get_neuron(s->pool, u1);
    failed += check(near2(n->weights, 1.0f - 0.5f * h1, 0.0f), "rank 1 moves by exp(-1/lambda)");
    n = nnpool_get_neuron(s->pool, u2);
    failed += check(near2(n->weights, 3.0f - 1.5f * h2, 0.0f), "rank 2 moves by exp(-2/lambda)");
    failed += check(s->pool->degrees[u0] == 0 && s->pool->degrees[u1] == 0, "Neural Gas does not add edges");
    failed += check(m->observe_count == 1, "observe_count advances");

    failed += check(ngas_save(m, META_PATH, BIN_PATH) == 0, "ngas_save");
    loaded = ngas_load(META_PATH, BIN_PATH, &net2);
    failed += check(loaded != NULL && net2 != NULL, "ngas_load");
    if (loaded) {
        neuron_t *a = nnpool_get_neuron(s->pool, u2);
        neuron_t *b = nnpool_get_neuron(net2->pool, ngas_unit_at(loaded, 2));
        failed += check(loaded->n_units == 3 && loaded->observe_count == 1, "loaded size and step");
        failed += check(fabsf(loaded->lambda_start - 1.0f) < 1e-6f, "loaded lambda_start");
        failed += check(ngas_unit_at(loaded, 0) == u0, "loaded unit ids");
        failed += check(near2(b->weights, a->weights[0], a->weights[1]), "loaded prototype matches");
        ngas_destroy(loaded);
        sonn_destroy(net2);
    }

    ngas_destroy(m);
    sonn_destroy(s);

    /* lambda_start <= 0 means one rank-scale per unit. */
    s = sonn_create(2, 1, 16, 4, GELU);
    m = ngas_create(s, 5, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    failed += check(m != NULL && m->n_units == 5, "default create");
    if (m) {
        failed += check(m->lambda_start == 5.0f, "default lambda_start is n_units");
        failed += check(m->eps_start == NGAS_DEFAULT_EPS_START, "default epsilon");
        failed += check(m->n_steps == NGAS_DEFAULT_N_STEPS, "default horizon");
        ngas_destroy(m);
    }
    sonn_destroy(s);

    if (failed) {
        printf("test_ngas: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_ngas: PASS\n\n");
    return 0;
}
