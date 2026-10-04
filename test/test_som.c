/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* One Kohonen step on a 2x2 grid with a fixed schedule, then a save/load round trip. */

#include <serialcore/sonn/som.h>

#include <math.h>
#include <stdio.h>

static const char *META_PATH = "build/test_som_model.json";
static const char *BIN_PATH  = "build/test_som_model.bin";

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
    const float x[2] = {0.0f, 0.0f};
    float h1, h2;
    int failed = 0;
    int id00, id01, id10, id11;
    int bmu;
    sonn_t *net2 = NULL;
    som_t *loaded;
    neuron_t *n;

    sonn_t *s = sonn_create(2, 1, 16, 4, GELU);
    som_t *m = som_create(s, 2, 2, /*n_steps=*/8,
                          /*eps*/ 0.5f, 0.5f,
                          /*sigma*/ 1.0f, 1.0f);
    failed += check(s != NULL && m != NULL, "som_create 2x2");
    if (!m) {
        sonn_destroy(s);
        printf("test_som: FAIL (%d assertions broken)\n", failed);
        return 1;
    }

    failed += check(m->n_units == 4, "grid has 4 units");
    failed += check(s->current_neurons == s->in_count + s->out_count + 4, "only the grid was added");
    failed += check(fabsf(som_epsilon(m) - 0.5f) < 1e-6f, "step 0 uses eps_start");
    failed += check(fabsf(som_sigma(m) - 1.0f) < 1e-6f, "step 0 uses sigma_start");

    id00 = som_unit_at(m, 0, 0);
    id01 = som_unit_at(m, 0, 1);
    id10 = som_unit_at(m, 1, 0);
    id11 = som_unit_at(m, 1, 1);
    write_at(s, id00, 0.0f, 0.0f);
    write_at(s, id01, 0.0f, 10.0f);
    write_at(s, id10, 10.0f, 0.0f);
    write_at(s, id11, 10.0f, 10.0f);

    bmu = som_find_bmu(m, x);
    failed += check(bmu == id00, "BMU is the origin cell");
    failed += check(som_observe(m, x) == 0, "som_observe");

    h1 = expf(-0.5f);
    h2 = expf(-1.0f);
    n = nnpool_get_neuron(s->pool, id00);
    failed += check(near2(n->weights, 0.0f, 0.0f), "BMU stays on the sample");
    n = nnpool_get_neuron(s->pool, id01);
    failed += check(near2(n->weights, 0.0f, 10.0f - 5.0f * h1), "edge neighbor moves by exp(-1/2)");
    n = nnpool_get_neuron(s->pool, id10);
    failed += check(near2(n->weights, 10.0f - 5.0f * h1, 0.0f), "other edge neighbor matches");
    n = nnpool_get_neuron(s->pool, id11);
    failed += check(near2(n->weights, 10.0f - 5.0f * h2, 10.0f - 5.0f * h2), "diagonal uses grid distance 2");
    failed += check(s->pool->degrees[id00] == 0, "SOM does not add edges");
    failed += check(m->observe_count == 1, "observe_count advances");

    failed += check(som_save(m, META_PATH, BIN_PATH) == 0, "som_save");
    loaded = som_load(META_PATH, BIN_PATH, &net2);
    failed += check(loaded != NULL && net2 != NULL, "som_load");
    if (loaded) {
        neuron_t *a;
        neuron_t *b;
        failed += check(loaded->rows == 2 && loaded->cols == 2, "loaded grid shape");
        failed += check(loaded->observe_count == 1, "loaded observe_count");
        failed += check(som_unit_at(loaded, 0, 1) == id01, "loaded cell ids");
        a = nnpool_get_neuron(s->pool, id11);
        b = nnpool_get_neuron(net2->pool, som_unit_at(loaded, 1, 1));
        failed += check(near2(b->weights, a->weights[0], a->weights[1]), "loaded prototype matches");
        som_destroy(loaded);
        sonn_destroy(net2);
    }

    som_destroy(m);
    sonn_destroy(s);

    if (failed) {
        printf("test_som: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_som: PASS\n\n");
    return 0;
}
