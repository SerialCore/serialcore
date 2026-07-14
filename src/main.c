/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/network.h>
#include <serialcore/sonn/layer.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* XOR problem */
static const float X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
static const float Y[4]     = { 0 ,  1 ,  1 ,  0  };

int main(void)
{
    /* 4-layer feedforward for XOR:
     *   layer 0: 2 inputs
     *   layer 1: 20 hidden (GELU)
     *   layer 2: 20 hidden (GELU)
     *   layer 3: 1 output (GELU or linear-ish)
     */
    int layer_sizes[4] = {2, 20, 20, 1};
    activaton_t layer_acts[4] = {GELU, GELU, GELU, GELU};

    /* Create a network with a pool of 256 slots, each with 128-dim weight vectors.
     * The actual fan-in is defined by the edges we add, not by pool input_dim.
     */
    network_t *net = network_create(256, 64);
    if (!net) {
        fprintf(stderr, "Failed to create network\n");
        return 1;
    }

    if (layer_build_ffnn(net, layer_sizes, 4, layer_acts) != 0) {
        fprintf(stderr, "Failed to build layered topology\n");
        network_destroy(net);
        return 1;
    }

    printf("Built 2-20-20-1 feedforward network on SONN (XOR task)\n");

    float lr = 0.8f;
    int epochs = 3000;

    for (int e = 0; e < epochs; e++) {
        float mse = 0.0f;

        for (int i = 0; i < 4; i++) {
            float pred[1];
            layer_ffnn_forward(net, X[i], pred);

            float err = Y[i] - pred[0];
            mse += err * err;

            layer_ffnn_train_step(net, X[i], Y[i], lr);
        }

        if ((e % 300) == 0 || e == epochs-1) {
            printf("epoch %4d   mse = %.6f\n", e, mse / 4.0f);
        }
    }

    printf("\nFinal test:\n");
    for (int i = 0; i < 4; i++) {
        float pred[1];
        layer_ffnn_forward(net, X[i], pred);
        printf("  XOR(%g,%g) = %.3f   (target %g)\n",
               X[i][0], X[i][1], pred[0], Y[i]);
    }

    network_destroy(net);
    return 0;
}
