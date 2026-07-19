/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_GEMM
#define SERIALCORE_FFNN_GEMM

#include <serialcore/sonn/activaton.h>

/*
 * Tiny BLAS + GEMM helpers shared across all ffnn layer implementations.
 * Implemented in src/ffnn/gemm.c so this header is the 1:1 declaration
 * surface for that file.
 *
 * Notation in `ffnn_gemm`:
 *   C[M,N] := alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
 *   TA/TB:  0 -> identity, 1 -> transpose
 *   lda / ldb / ldc: row stride for the *untransposed* matrix of A/B/C
 *     (matches darknet's convention; an unrolled gemm_tn uses A[k*lda+i]).
 */

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
               const float *A, int lda,
               const float *B, int ldb,
               float BETA,
               float *C, int ldc);

void gemm_fill (int n, float val, float *x, int incx);
void gemm_axpy (int n, float a, const float *x, int incx,
                    float *y, int incy);
void gemm_scal (int n, float a, float *x, int incx);
void gemm_copy (int n, const float *x, int incx, float *y, int incy);

/* Add bias to every row of a [batch, outputs] activation matrix. */
void gemm_add_bias    (const float *biases, float *out, int batch, int outputs);
/* Sum-reduce bias gradients across the batch. */
void gemm_backward_bias(float *bias_updates, const float *delta, int batch, int outputs);

void gemm_activate_array (float *x, int n, activaton_t a);
/* Multiply delta[i] by activaton'(x[i]) in place. Darknet uses the
 * post-activation output (x here) for the gradient argument because the
 * inlined derivatives in <serialcore/sonn/activaton.h> are written to
 * accept the pre-activation sum; for the activation set serialcore
 * supports today this distinction is well-defined (see the static inline
 * gf_* functions). */
void gemm_gradient_array (const float *pre_act, int n, activaton_t a, float *delta);

#endif