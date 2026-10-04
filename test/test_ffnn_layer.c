/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Layer ops: unimplemented types fail, compile is closed, identity is linear,
 * and the loss callback is what writes the output delta. */

#include <serialcore/ffnn/ffnn.h>

#include <stdio.h>
#include <string.h>

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

static void zero_loss(const float *output, const float *target, float *delta, int n, void *userdata)
{
    (void)output;
    (void)target;
    (void)userdata;
    for (int i = 0; i < n; i++) delta[i] = 0.0f;
}

int main(void)
{
    int failed = 0;
    ffnn_network_t *net;
    float x = 0.5f;
    float y = 0.0f;
    float before[2];

    net = ffnn_create(2, 1, 0.1f, 0.0f, 0.0f);
    failed += check(net != NULL, "ffnn_create");
    failed += check(ffnn_add_layer(net, 2, 2, FFNN_CONVOLUTIONAL, ReLU) != 0, "convolution is rejected");
    failed += check(ffnn_add_layer(net, 2, 2, FFNN_MAXPOOL, ReLU) != 0, "maxpool is rejected");
    failed += check(ffnn_add_layer(net, 2, 2, FFNN_SOFTMAX, ReLU) != 0, "softmax is rejected");
    failed += check(net->n == 0, "rejected layers are not appended");
    ffnn_destroy(net);

    net = ffnn_create(1, 1, 0.1f, 0.0f, 0.0f);
    failed += check(ffnn_add_layer(net, 1, 1, FFNN_DENSE, Identity) == 0, "identity dense layer");
    failed += check(ffnn_compile(net) == 0, "compile");
    failed += check(ffnn_add_layer(net, 1, 1, FFNN_DENSE, Identity) != 0, "add after compile fails");
    net->pool->params[0] = 2.0f;
    net->pool->params[1] = -0.25f;
    failed += check(ffnn_predict(net, &x, &y) == 0, "predict");
    failed += check(y > 0.749f && y < 0.751f, "identity computes 2*x - 0.25");

    before[0] = net->pool->params[0];
    before[1] = net->pool->params[1];
    ffnn_set_loss(net, zero_loss, NULL);
    failed += check(ffnn_train_step(net, &x, &x) == 0, "train step with zero loss");
    failed += check(net->pool->params[0] == before[0] && net->pool->params[1] == before[1],
                    "zero loss leaves parameters unchanged");
    ffnn_destroy(net);

    if (failed) {
        printf("test_ffnn_layer: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_ffnn_layer: PASS\n\n");
    return 0;
}
