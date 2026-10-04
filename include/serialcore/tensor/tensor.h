/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_TENSOR_TENSOR
#define SERIALCORE_TENSOR_TENSOR

#include <stddef.h>

/* Rank cap covers N,C,H,W. A tensor is a float buffer plus shape and stride.
 * Element (i0, i1, ...) lives at data[sum_d index[d] * stride[d]]. */
#define TENSOR_MAX_RANK 4

typedef struct tensor {
    float *data;
    int    rank;
    int    shape[TENSOR_MAX_RANK];
    int    stride[TENSOR_MAX_RANK];
    int    owns;   /* 1: tensor_destroy frees data. The struct itself is the caller's. */
} tensor_t;

/* Row-major contiguous buffer. shape[i] > 0, 1 <= rank <= TENSOR_MAX_RANK.
 * The element count must fit in int. On failure t is left unchanged. */
int tensor_create(tensor_t *t, const int *shape, int rank);

/* Borrow data. owns is 0. stride[i] == 0 is rejected. A negative stride is a
 * reverse axis. On failure t is left unchanged. */
int tensor_view(tensor_t *t, float *data, const int *shape, const int *stride, int rank);

/* Free data when owns is 1. A view leaves the borrowed buffer in place. */
void tensor_destroy(tensor_t *t);

/* Write value at every logical element, following stride. */
int tensor_fill(tensor_t *t, float value);

/* Copy logical elements. Ranks and shapes must match; strides may differ. */
int tensor_copy(const tensor_t *src, tensor_t *dst);

/* C = A * B for rank-2 tensors, C[M,N], A[M,K], B[K,N]. Stride supplies any
 * transpose, so a transposed argument is a view. C is overwritten. C must not
 * alias A or B. */
int tensor_matmul(const tensor_t *a, const tensor_t *b, tensor_t *c);

/* Elementwise C = A + B and C = A * B. Ranks and shapes must match; strides
 * may differ. No broadcasting. C is overwritten. */
int tensor_add(const tensor_t *a, const tensor_t *b, tensor_t *c);
int tensor_mul(const tensor_t *a, const tensor_t *b, tensor_t *c);

/* Sum src along one axis into dst, dropping that axis. src->rank >= 2, and
 * dst->rank == src->rank - 1 with the remaining shape. dst is overwritten.
 * A shape mismatch leaves dst unchanged. */
int tensor_sum(const tensor_t *src, int axis, tensor_t *dst);

/* Offset of one logical element. index has t->rank entries. */
static inline ptrdiff_t tensor_offset(const tensor_t *t, const int *index)
{
    ptrdiff_t off = 0;
    if (!t || !index) return 0;
    for (int d = 0; d < t->rank; ++d)
        off += (ptrdiff_t)index[d] * (ptrdiff_t)t->stride[d];
    return off;
}

#endif
