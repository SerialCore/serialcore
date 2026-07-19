/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_GEMM
#define SERIALCORE_FFNN_GEMM

#include <serialcore/sonn/activaton.h>

/*
 * FFNN GEMM + tiny BLAS helpers.
 *
 * The kernel is the classic ikj triple loop used by darknet's gemm_nn; it is
 * cache-friendly enough that the Dense layer in the XOR test never becomes a
 * bottleneck. OpenMP is enabled when -DOPENMP is set on the compile line, the
 * same flag the rest of serialcore already respects.
 *
 * Notation:
 *   ffnn_gemm(TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
 *
 *   C[M,N] := alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
 *
 *   TA / TB: 0 → no transpose (op is identity), 1 → op is A^T / B^T.
 *
 * Leading dimensions follow darknet's convention: lda is the stride for the
 * *untransposed* matrix, so the inner access patterns are
 *   !TA: A[i*lda + k]    TA: A[k*lda + i]
 *   !TB: B[k*ldb + j]    TB: B[j*ldb + k]
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