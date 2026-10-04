/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_DENSE
#define SERIALCORE_FFNN_DENSE

#include <serialcore/ffnn/ffnn.h>

/*
 * Dense layer.
 *
 * Forward:  y[b,o] = activaton( sum_i( x[b,i] * W[o,i] ) + bias[o] )
 *   GEMM: C[batch, outputs] = X[batch, inputs] * W^T, then add bias.
 *
 * Backward, after the loss has written delta and this layer multiplies by f'(z):
 *   weight_updates[o,i] += sum_b( dy[b,o] * x[b,i] )
 *   bias_updates[o]      += sum_b( dy[b,o] )
 *   dx[b,i]              += sum_o( dy[b,o] * W[o,i] )
 *
 * Parameter block, bound at compile time: weights[outputs * inputs], then biases[outputs].
 */

const ffnn_layer_ops_t *dense_layer_ops(void);

/* Allocate the dense workspace. Parameters stay unbound until ffnn_compile. */
int dense_prepare(ffnn_layer_t *l);

#endif
