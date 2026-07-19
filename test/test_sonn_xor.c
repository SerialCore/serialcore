/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit test: XOR trained on the SONN core (FFNN-as-special-case on top of the
 * memory pool), mirroring src/main.c but with assertions at the end.
 */

#include <serialcore/sonn/network.h>
#include <serialcore/sonn/layer.h>
#include <serialcore/math/xoshiross.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* XOR problem */
static const float X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
static const float Y[4]     = { 0 ,  1 ,  1 ,  0  };

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

int main(void)
{
    /* Weight init in ffnn_make_dense_layer consumes the project-wide RNG,
     * so seed it deterministically before any layer is allocated. */
    xoshiro_seed(0xC0FFEEULL);

    int layer_sizes[4] = {2, 20, 20, 1};
    activaton_t layer_acts[4] = {GELU, GELU, GELU, GELU};

    network_t *net = network_create(256, 128, 64);
    if (!net) { fprintf(stderr, "network_create failed\n"); return 1; }

    if (layer_build_ffnn(net, layer_sizes, 4, layer_acts) != 0) {
        fprintf(stderr, "layer_build_ffnn failed\n");
        network_destroy(net);
        return 1;
    }

    const float lr = 0.8f;
    const int epochs = 3000;

    float mse = 0.0f;
    for (int e = 0; e < epochs; e++) {
        mse = 0.0f;
        for (int i = 0; i < 4; i++) {
            float pred[1];
            layer_ffnn_forward(net, X[i], pred);
            float err = Y[i] - pred[0];
            mse += err * err;
            layer_ffnn_train_step(net, X[i], Y[i], lr);
        }
        if ((e % 600) == 0 || e == epochs - 1) {
            printf("epoch %4d   mse = %.6f\n", e, mse / 4.0f);
        }
    }

    int failed = 0;

    /* Convergence: average squared error should be well under the naive
     * 0.25 baseline (always predicting 0.5). */
    failed += check(mse / 4.0f < 0.01f, "final MSE < 0.01");

    /* Each XOR pattern should be close to its target. */
    for (int i = 0; i < 4; i++) {
        float pred[1];
        layer_ffnn_forward(net, X[i], pred);
        char buf[64];
        snprintf(buf, sizeof(buf),
                 "XOR(%g,%g) = %.3f (target %g)",
                 X[i][0], X[i][1], pred[0], Y[i]);
        failed += check(fabsf(pred[0] - Y[i]) < 0.15f, buf);
    }

    network_destroy(net);

    if (failed) {
        printf("test_sonn_xor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_sonn_xor: PASS\n\n");
    return 0;
}