/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_DENSE
#define SERIALCORE_FFNN_DENSE

#include <serialcore/ffnn/ffnn.h>

/*
 * Dense (a.k.a. connected / fully-connected) layer.
 *
 * Forward:   y[b,o] = activaton( sum_i( x[b,i] * W[o,i] ) + bias[o] )
 *
 *   Equivalent GEMM form (mirrors darknet's connected_layer.c):
 *     M = batch                 N = outputs                 K = inputs
 *     A = x      [batch * inputs]   laid out as net->input
 *     B = W^T    [outputs * inputs] stored row-major per output, gemm with TB=1
 *     C = pre-activation output [batch * outputs]
 *
 * Backward:  given dy[b,o] = (target-chain) * activaton'(z[b,o])
 *
 *   weight_updates [o,i] += sum_b( dy[b,o] * x[b,i] )
 *     →  GEMM(A=dy^T [outputs*batch], B=x [batch*inputs], C=Wup [outputs*inputs])
 *   bias_updates    [o]   += sum_b( dy[b,o] )
 *   dx[b,i]         += sum_o( dy[b,o] * W[o,i] )
 *     →  GEMM(A=dy [batch*outputs], B=W [outputs*inputs] (no transpose),
 *             C=net->delta [batch*inputs])
 */

/* Dense layer vtable entries. 
 * Exposed because ffnn_make_layer wires them into the freshly-built ffnn_layer_t. */
void dense_forward(ffnn_layer_t *l, ffnn_network_t *net);
void dense_backward(ffnn_layer_t *l, ffnn_network_t *net);
void dense_update(ffnn_layer_t *l, float lr, float momentum, float decay, int batch);

#endif