/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit test: XOR trained on the new dense-matrix FFNN core
 * (src/ffnn), used as the canonical smoke test that the darknet-style
 * vtable + GEMM forward/backward/update loop is wired up correctly.
 *
 * Same network shape as the SONN demo: 2-20-20-1, all GELU activations.
 */

#include <serialcore/ffnn/ffnn.h>
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
    /* Weight init in ffnn_make_layer consumes the project-wide RNG,
     * so seed it deterministically before any layer is allocated. */
    xoshiro_seed(0xC0FFEEULL);

    /* Hyperparameters chosen to mirror src/main.c's SONN demo. */
    const int   batch    = 1;
    const float lr       = 0.10f;
    const float momentum = 0.90f;
    const float decay    = 0.00f;
    const int   epochs   = 3000;

    ffnn_network_t *net = ffnn_create(2, batch, lr, momentum, decay);
    if (!net) { fprintf(stderr, "ffnn_network_create failed\n"); return 1; }

    /* 2 → 20 → 20 → 1, GELU throughout. */
    if (ffnn_add_layer(net, 2,  20, FFNN_DENSE, GELU) != 0 ||
        ffnn_add_layer(net, 20, 20, FFNN_DENSE, GELU) != 0 ||
        ffnn_add_layer(net, 20, 1,  FFNN_DENSE, GELU) != 0) {
        fprintf(stderr, "ffnn_network_add_layer failed\n");
        ffnn_destroy(net);
        return 1;
    }

    float mse = 0.0f;
    for (int e = 0; e < epochs; e++) {
        mse = 0.0f;
        for (int i = 0; i < 4; i++) {
            float pred[1];
            ffnn_predict(net, X[i], pred);
            float err = Y[i] - pred[0];
            mse += err * err;
            ffnn_train_step(net, X[i], &Y[i]);
        }
        if ((e % 600) == 0 || e == epochs - 1) {
            printf("epoch %4d   mse = %.6f\n", e, mse / 4.0f);
        }
    }

    int failed = 0;

    failed += check(mse / 4.0f < 0.01f, "final MSE < 0.01");

    for (int i = 0; i < 4; i++) {
        float pred[1];
        ffnn_predict(net, X[i], pred);
        char buf[64];
        snprintf(buf, sizeof(buf),
                 "XOR(%g,%g) = %.3f (target %g)",
                 X[i][0], X[i][1], pred[0], Y[i]);
        failed += check(fabsf(pred[0] - Y[i]) < 0.15f, buf);
    }

    ffnn_destroy(net);

    if (failed) {
        printf("test_ffnn_xor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_ffnn_xor: PASS\n\n");
    return 0;
}