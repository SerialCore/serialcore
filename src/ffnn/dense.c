/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Dense / fully-connected layer implementation.
 *
 * Forward (one shot of M=batch rows):
 *   C[b,o]  += sum_i A[b,i] * B^T[o,i]      (gemm 0,1)
 *   C[b,o]  += bias[o]
 *   out[b,o] = activaton(C[b,o])
 *
 * Backward:
 *   delta[b,o] *= activaton'(pre_act[b,o])            (gradient at z)
 *   bias_updates[o] += sum_b delta[b,o]
 *   weight_updates[o*inputs + i] += sum_b delta[b,o] * in[b,i]
 *   net->delta[b,i] += sum_o delta[b,o] * W[o*inputs + i]
 *
 * The pre-activation sum `pre_act` is what the next backward step consumes
 * (and what gradient_func expects as its argument — see <serialcore/sonn/activaton.h>).
 */

#include <serialcore/ffnn/dense.h>
#include <serialcore/ffnn/ffnn.h>
#include <serialcore/ffnn/gemm.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void dense_forward(ffnn_layer_t *l, ffnn_network_t *net)
{
    /* C [M=batch, N=outputs] = A [batch, inputs] * B^T [outputs, inputs]
     *   → gemm(TA=0, TB=1, M, N, K=inputs). */
    gemm(0, 1, l->batch, l->outputs, l->inputs,
              1.0f, net->input, l->inputs,
              l->weights, l->inputs,
              0.0f, l->pre_act, l->outputs);

    gemm_add_bias(l->biases, l->pre_act, l->batch, l->outputs);

    /* Copy z into output then activate in place. */
    for (int i = 0; i < l->batch * l->outputs; ++i) {
        l->output[i] = l->pre_act[i];
    }
    gemm_activate_array(l->output, l->batch * l->outputs, l->activation);

    /* Cache input for backward when training. */
    if (net && net->train && net->input && l->input_snapshot) {
        for (int i = 0; i < l->batch * l->inputs; ++i) {
            l->input_snapshot[i] = net->input[i];
        }
    }
}

void dense_backward(ffnn_layer_t *l, ffnn_network_t *net)
{
    /* Apply activation derivative at z (pre_act). `l->delta` has been set
     * upstream by the network (output layer gets target-output; hidden
     * layers get the chained delta from the next layer). */
    gemm_gradient_array(l->pre_act, l->batch * l->outputs, l->activation, l->delta);

    /* bias_updates += sum_b delta[b,*] (per output) */
    gemm_backward_bias(l->bias_updates, l->delta, l->batch, l->outputs);

    /* weight_updates [outputs, inputs] += delta^T [outputs, batch] *
     *                                        in [batch, inputs]
     *   → gemm(TA=1, TB=0, M=outputs, N=inputs, K=batch). */
    gemm(1, 0, l->outputs, l->inputs, l->batch,
              1.0f, l->delta, l->outputs,
              l->input_snapshot, l->inputs,
              1.0f, l->weight_updates, l->inputs);

    /* net->delta [batch, inputs] += delta [batch, outputs] *
     *                                 W [outputs, inputs]
     *   → gemm(TA=0, TB=0, M=batch, N=inputs, K=outputs). */
    if (net && net->delta) {
        gemm(0, 0, l->batch, l->inputs, l->outputs,
                  1.0f, l->delta, l->outputs,
                  l->weights, l->inputs,
                  1.0f, net->delta, l->inputs);
    }
}

void dense_update(ffnn_layer_t *l, float lr, float momentum, float decay, int batch)
{
    /* bias:    b -= (lr/batch) * bias_updates;   bias_updates *= momentum
     * weights: w -= (lr/batch) * (weight_updates + decay * batch * w)
     *          weight_updates *= momentum */
    const float scale = lr / (float)batch;

    for (int o = 0; o < l->outputs; ++o) {
        l->biases[o] += scale * l->bias_updates[o];
        l->bias_updates[o] *= momentum;
    }

    const int nw = l->outputs * l->inputs;
    for (int i = 0; i < nw; ++i) {
        const float g = l->weight_updates[i] - decay * batch * l->weights[i];
        l->weights[i] += scale * g;
        l->weight_updates[i] = momentum * l->weight_updates[i];
    }
}