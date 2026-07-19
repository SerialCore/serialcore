/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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

#include <serialcore/ffnn/gemm.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>

static void gemm_nn(int M, int N, int K, float ALPHA,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = ALPHA * A[i * lda + k];
            const float *brow = B + k * ldb;
            float *crow = C + i * ldc;
            for (int j = 0; j < N; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

/* A[M,K] (no transpose), B[N,K] transposed → B^T is [K,N] effectively */
static void gemm_nt(int M, int N, int K, float ALPHA,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float *arow = A + i * lda;
            const float *brow = B + j * ldb;
            for (int k = 0; k < K; ++k) {
                sum += arow[k] * brow[k];
            }
            C[i * ldc + j] += ALPHA * sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, float ALPHA,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = ALPHA * A[k * lda + i];
            const float *brow = B + k * ldb;
            float *crow = C + i * ldc;
            for (int j = 0; j < N; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, float ALPHA,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float *C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k * lda + i] * B[j * ldb + k];
            }
            C[i * ldc + j] += ALPHA * sum;
        }
    }
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
               const float *A, int lda,
               const float *B, int ldb,
               float BETA,
               float *C, int ldc)
{
    if (BETA == 0.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (BETA != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] *= BETA;
            }
        }
    }

    if (!TA && !TB)      gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)  gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)  gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else                 gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void gemm_fill(int n, float val, float *x, int incx)
{
    for (int i = 0; i < n; ++i) x[i * incx] = val;
}

void gemm_axpy(int n, float a, const float *x, int incx,
                  float *y, int incy)
{
    for (int i = 0; i < n; ++i) y[i * incy] += a * x[i * incx];
}

void gemm_scal(int n, float a, float *x, int incx)
{
    for (int i = 0; i < n; ++i) x[i * incx] *= a;
}

void gemm_copy(int n, const float *x, int incx, float *y, int incy)
{
    for (int i = 0; i < n; ++i) y[i * incy] = x[i * incx];
}

void gemm_add_bias(const float *biases, float *out, int batch, int outputs)
{
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < outputs; ++o) {
            out[b * outputs + o] += biases[o];
        }
    }
}

void gemm_backward_bias(float *bias_updates, const float *delta, int batch, int outputs)
{
    for (int o = 0; o < outputs; ++o) {
        float acc = 0.0f;
        for (int b = 0; b < batch; ++b) {
            acc += delta[b * outputs + o];
        }
        bias_updates[o] += acc;
    }
}

void gemm_activate_array(float *x, int n, activaton_t a)
{
    for (int i = 0; i < n; ++i) {
        x[i] = activaton_func(x[i], a);
    }
}

void gemm_gradient_array(const float *activation_out, int n, activaton_t a, float *delta)
{
    for (int i = 0; i < n; ++i) {
        delta[i] *= gradient_func(activation_out[i], a);
    }
}