/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/tensor/tensor.h>

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int check_shape(const int *shape, int rank, int *numel)
{
    long long n = 1;

    if (!shape || rank < 1 || rank > TENSOR_MAX_RANK) return -1;
    for (int i = 0; i < rank; ++i) {
        if (shape[i] <= 0 || n > (long long)INT_MAX / shape[i]) return -1;
        n *= shape[i];
    }
    if (numel) *numel = (int)n;
    return 0;
}

/* Product of the axes to the right of i. Fits in int when the shape does. */
static void contiguous_stride(const int *shape, int rank, int *stride)
{
    long long step = 1;

    for (int i = rank - 1; i >= 0; --i) {
        stride[i] = (int)step;
        step *= shape[i];
    }
}

/* Every logical index must land at a signed-int offset from data. */
static int check_stride(const int *shape, const int *stride, int rank)
{
    long long span = 0;

    if (!stride) return -1;
    for (int i = 0; i < rank; ++i) {
        long long term;
        long long mag;

        if (stride[i] == 0) return -1;
        term = (long long)(shape[i] - 1) * (long long)stride[i];
        if (term > INT_MAX || term < INT_MIN) return -1;
        mag = term < 0 ? -term : term;
        if (span > (long long)INT_MAX - mag) return -1;
        span += mag;
    }
    return 0;
}

static ptrdiff_t offset_at(const tensor_t *t, const int *idx)
{
    ptrdiff_t off = 0;

    for (int d = 0; d < t->rank; ++d)
        off += (ptrdiff_t)idx[d] * (ptrdiff_t)t->stride[d];
    return off;
}

static void odometer_next(int *idx, const int *shape, int rank)
{
    for (int d = rank - 1; d >= 0; --d) {
        idx[d] += 1;
        if (idx[d] < shape[d]) return;
        idx[d] = 0;
    }
}

int tensor_create(tensor_t *t, const int *shape, int rank)
{
    int stride[TENSOR_MAX_RANK];
    int numel;
    float *data;

    if (!t || check_shape(shape, rank, &numel) != 0) return -1;
    contiguous_stride(shape, rank, stride);
    data = (float *)calloc((size_t)numel, sizeof(float));
    if (!data) return -1;

    memset(t, 0, sizeof(*t));
    t->data = data;
    t->rank = rank;
    t->owns = 1;
    for (int i = 0; i < rank; ++i) {
        t->shape[i] = shape[i];
        t->stride[i] = stride[i];
    }
    return 0;
}

int tensor_view(tensor_t *t, float *data, const int *shape, const int *stride, int rank)
{
    if (!t || !data || check_shape(shape, rank, NULL) != 0) return -1;
    if (check_stride(shape, stride, rank) != 0) return -1;

    memset(t, 0, sizeof(*t));
    t->data = data;
    t->rank = rank;
    t->owns = 0;
    for (int i = 0; i < rank; ++i) {
        t->shape[i] = shape[i];
        t->stride[i] = stride[i];
    }
    return 0;
}

void tensor_destroy(tensor_t *t)
{
    if (!t) return;
    if (t->owns) free(t->data);
    t->data = NULL;
    t->owns = 0;
    t->rank = 0;
}

int tensor_fill(tensor_t *t, float value)
{
    int idx[TENSOR_MAX_RANK] = {0};
    int numel;

    if (!t || !t->data || check_shape(t->shape, t->rank, &numel) != 0) return -1;
    if (check_stride(t->shape, t->stride, t->rank) != 0) return -1;
    for (int e = 0; e < numel; ++e) {
        t->data[offset_at(t, idx)] = value;
        odometer_next(idx, t->shape, t->rank);
    }
    return 0;
}

int tensor_copy(const tensor_t *src, tensor_t *dst)
{
    int idx[TENSOR_MAX_RANK] = {0};
    int numel;

    if (!src || !dst || !src->data || !dst->data) return -1;
    if (src->rank != dst->rank) return -1;
    for (int i = 0; i < src->rank; ++i) {
        if (src->shape[i] != dst->shape[i]) return -1;
    }
    if (check_shape(src->shape, src->rank, &numel) != 0) return -1;
    if (check_stride(src->shape, src->stride, src->rank) != 0) return -1;
    if (check_stride(dst->shape, dst->stride, dst->rank) != 0) return -1;

    for (int e = 0; e < numel; ++e) {
        dst->data[offset_at(dst, idx)] = src->data[offset_at(src, idx)];
        odometer_next(idx, src->shape, src->rank);
    }
    return 0;
}

static int same_shape(const tensor_t *a, const tensor_t *b)
{
    if (!a || !b || a->rank != b->rank) return 0;
    for (int i = 0; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

static int tensor_usable(const tensor_t *t)
{
    if (!t || !t->data) return 0;
    if (check_shape(t->shape, t->rank, NULL) != 0) return 0;
    if (check_stride(t->shape, t->stride, t->rank) != 0) return 0;
    return 1;
}

static int tensor_binop(const tensor_t *a, const tensor_t *b, tensor_t *c, int multiply)
{
    int idx[TENSOR_MAX_RANK] = {0};
    int numel;

    if (!tensor_usable(a) || !tensor_usable(b) || !tensor_usable(c)) return -1;
    if (!same_shape(a, b) || !same_shape(a, c)) return -1;
    if (check_shape(a->shape, a->rank, &numel) != 0) return -1;

    for (int e = 0; e < numel; ++e) {
        float av = a->data[offset_at(a, idx)];
        float bv = b->data[offset_at(b, idx)];
        c->data[offset_at(c, idx)] = multiply ? av * bv : av + bv;
        odometer_next(idx, a->shape, a->rank);
    }
    return 0;
}

int tensor_add(const tensor_t *a, const tensor_t *b, tensor_t *c)
{
    return tensor_binop(a, b, c, 0);
}

int tensor_mul(const tensor_t *a, const tensor_t *b, tensor_t *c)
{
    return tensor_binop(a, b, c, 1);
}

int tensor_sum(const tensor_t *src, int axis, tensor_t *dst)
{
    int idx[TENSOR_MAX_RANK] = {0};
    int reduced[TENSOR_MAX_RANK];
    int numel;
    int d = 0;

    if (!tensor_usable(src) || !tensor_usable(dst)) return -1;
    if (src->rank < 2 || axis < 0 || axis >= src->rank) return -1;
    if (dst->rank != src->rank - 1) return -1;
    for (int i = 0; i < src->rank; ++i) {
        if (i == axis) continue;
        if (dst->shape[d] != src->shape[i]) return -1;
        d += 1;
    }
    if (tensor_fill(dst, 0.0f) != 0) return -1;
    if (check_shape(src->shape, src->rank, &numel) != 0) return -1;

    for (int e = 0; e < numel; ++e) {
        int r = 0;
        for (int i = 0; i < src->rank; ++i) {
            if (i == axis) continue;
            reduced[r] = idx[i];
            r += 1;
        }
        dst->data[offset_at(dst, reduced)] += src->data[offset_at(src, idx)];
        odometer_next(idx, src->shape, src->rank);
    }
    return 0;
}

int tensor_matmul(const tensor_t *a, const tensor_t *b, tensor_t *c)
{
    int m, k, n;

    if (!a || !b || !c || !a->data || !b->data || !c->data) return -1;
    if (a->rank != 2 || b->rank != 2 || c->rank != 2) return -1;
    m = a->shape[0];
    k = a->shape[1];
    n = b->shape[1];
    if (k <= 0 || m <= 0 || n <= 0) return -1;
    if (b->shape[0] != k || c->shape[0] != m || c->shape[1] != n) return -1;
    if (tensor_fill(c, 0.0f) != 0) return -1;

    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            int ia[2] = {i, p};
            float av = a->data[offset_at(a, ia)];
            for (int j = 0; j < n; ++j) {
                int ib[2] = {p, j};
                int ic[2] = {i, j};
                c->data[offset_at(c, ic)] += av * b->data[offset_at(b, ib)];
            }
        }
    }
    return 0;
}
