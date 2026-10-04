/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Stride is the address rule: a transpose is a view, and matmul reads it. */

#include <serialcore/tensor/tensor.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int check(int passed, const char *msg)
{
    printf("  [%s] %s\n", passed ? "PASS" : "FAIL", msg);
    return passed ? 0 : 1;
}

static float at2(const tensor_t *t, int i, int j)
{
    return t->data[(ptrdiff_t)i * t->stride[0] + (ptrdiff_t)j * t->stride[1]];
}

int main(void)
{
    int failed = 0;
    int shape2[2] = {2, 2};
    int huge[2] = {INT_MAX, 2};
    int zero_dim[2] = {2, 0};
    tensor_t blank = {0};
    tensor_t a = {0};
    tensor_t view = {0};
    tensor_t copied = {0};
    float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int transpose_stride[2] = {1, 2};

    failed += check(tensor_create(NULL, shape2, 2) != 0, "create rejects NULL");
    failed += check(tensor_create(&blank, shape2, 0) != 0, "create rejects rank 0");
    failed += check(tensor_create(&blank, shape2, 5) != 0, "create rejects rank 5");
    failed += check(tensor_create(&blank, zero_dim, 2) != 0, "create rejects a zero axis");
    failed += check(tensor_create(&blank, huge, 2) != 0, "create rejects a product that overflows int");
    failed += check(blank.data == NULL && blank.owns == 0, "failed create leaves the struct unchanged");

    failed += check(tensor_create(&a, shape2, 2) == 0, "create 2x2");
    failed += check(a.owns == 1, "create owns the buffer");
    failed += check(a.stride[0] == 2 && a.stride[1] == 1, "contiguous stride is row-major");
    a.data[0] = 1.0f;
    a.data[1] = 2.0f;
    a.data[2] = 3.0f;
    a.data[3] = 4.0f;

    failed += check(tensor_view(&view, buf, shape2, transpose_stride, 2) == 0, "transpose view");
    failed += check(view.owns == 0, "view does not own the buffer");
    failed += check(at2(&view, 0, 0) == 1.0f && at2(&view, 0, 1) == 3.0f
                    && at2(&view, 1, 0) == 2.0f && at2(&view, 1, 1) == 4.0f,
                    "transpose view reads [[1,3],[2,4]]");

    failed += check(tensor_create(&copied, shape2, 2) == 0, "create copy destination");
    failed += check(tensor_copy(&view, &copied) == 0, "copy follows stride");
    failed += check(copied.data[0] == 1.0f && copied.data[1] == 3.0f
                    && copied.data[2] == 2.0f && copied.data[3] == 4.0f,
                    "copied transpose is contiguous [[1,3],[2,4]]");

    failed += check(tensor_fill(&copied, 5.0f) == 0, "fill");
    failed += check(copied.data[0] == 5.0f && copied.data[1] == 5.0f
                    && copied.data[2] == 5.0f && copied.data[3] == 5.0f,
                    "fill writes every element");

    tensor_destroy(&view);
    failed += check(buf[0] == 1.0f && buf[1] == 2.0f && buf[2] == 3.0f && buf[3] == 4.0f,
                    "destroy of a view leaves the borrowed buffer");

    {
        /* A is 2x3, B is 3x2. C = [[58, 64], [139, 154]]. */
        int ashape[2] = {2, 3};
        int bshape[2] = {3, 2};
        int cshape[2] = {2, 2};
        int b_as_rows[2] = {2, 3};
        int b_view_stride[2] = {1, 3};
        float bt_storage[6] = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};
        tensor_t am = {0};
        tensor_t bm = {0};
        tensor_t cm = {0};
        tensor_t bview = {0};
        int bad_stride[2] = {0, 1};

        failed += check(tensor_create(&am, ashape, 2) == 0, "create A 2x3");
        failed += check(am.stride[0] == 3 && am.stride[1] == 1, "2x3 stride is [3,1]");
        failed += check(tensor_create(&bm, bshape, 2) == 0, "create B 3x2");
        failed += check(tensor_create(&cm, cshape, 2) == 0, "create C 2x2");
        am.data[0] = 1.0f; am.data[1] = 2.0f; am.data[2] = 3.0f;
        am.data[3] = 4.0f; am.data[4] = 5.0f; am.data[5] = 6.0f;
        bm.data[0] = 7.0f; bm.data[1] = 8.0f;
        bm.data[2] = 9.0f; bm.data[3] = 10.0f;
        bm.data[4] = 11.0f; bm.data[5] = 12.0f;
        tensor_fill(&cm, -1.0f);
        failed += check(tensor_matmul(&am, &a, &cm) != 0, "matmul rejects a K mismatch");
        failed += check(cm.data[0] == -1.0f, "rejected matmul leaves C unchanged");
        failed += check(tensor_matmul(&am, &bm, &cm) == 0, "matmul 2x3 by 3x2");
        failed += check(at2(&cm, 0, 0) == 58.0f && at2(&cm, 0, 1) == 64.0f
                        && at2(&cm, 1, 0) == 139.0f && at2(&cm, 1, 1) == 154.0f,
                        "C is [[58,64],[139,154]]");

        failed += check(tensor_view(&bview, bt_storage, bshape, b_view_stride, 2) == 0,
                        "B as a transpose view");
        failed += check(at2(&bview, 0, 0) == 7.0f && at2(&bview, 0, 1) == 8.0f
                        && at2(&bview, 2, 1) == 12.0f,
                        "transpose view restores B");
        tensor_fill(&cm, 0.0f);
        failed += check(tensor_matmul(&am, &bview, &cm) == 0, "matmul through a transpose view");
        failed += check(at2(&cm, 0, 0) == 58.0f && at2(&cm, 0, 1) == 64.0f
                        && at2(&cm, 1, 0) == 139.0f && at2(&cm, 1, 1) == 154.0f,
                        "transpose view yields the same C");
        failed += check(tensor_view(&bview, bt_storage, b_as_rows, bad_stride, 2) != 0,
                        "view rejects a zero stride");

        tensor_destroy(&bview);
        failed += check(bt_storage[0] == 7.0f && bt_storage[5] == 12.0f,
                        "destroy of the B view leaves its storage");
        tensor_destroy(&am);
        tensor_destroy(&bm);
        tensor_destroy(&cm);
    }

    {
        /* [[1,2,3],[4,5,6]] plus and times itself, including a stride view.
         * Sum of axis 0 is [5,7,9]. Sum of axis 1 is [6,15]. */
        int shape[2] = {2, 3};
        int along_row[1] = {3};
        int along_col[1] = {2};
        int bad_shape[1] = {2};
        int view_stride[2] = {1, 2};
        float storage[6] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
        tensor_t x = {0};
        tensor_t y = {0};
        tensor_t z = {0};
        tensor_t sum_axis0 = {0};
        tensor_t sum_axis1 = {0};
        tensor_t bad = {0};

        failed += check(tensor_create(&x, shape, 2) == 0, "create elementwise source");
        x.data[0] = 1.0f; x.data[1] = 2.0f; x.data[2] = 3.0f;
        x.data[3] = 4.0f; x.data[4] = 5.0f; x.data[5] = 6.0f;
        failed += check(tensor_view(&y, storage, shape, view_stride, 2) == 0,
                        "elementwise operand as a view");
        failed += check(tensor_create(&z, shape, 2) == 0, "create elementwise destination");
        tensor_fill(&z, -1.0f);
        failed += check(tensor_add(&x, &a, &z) != 0, "add rejects a shape mismatch");
        failed += check(z.data[0] == -1.0f, "rejected add leaves the destination unchanged");
        failed += check(tensor_add(&x, &y, &z) == 0, "add through a strided view");
        failed += check(z.data[0] == 2.0f && z.data[1] == 4.0f && z.data[2] == 6.0f
                        && z.data[3] == 8.0f && z.data[4] == 10.0f && z.data[5] == 12.0f,
                        "add doubles [[1,2,3],[4,5,6]]");
        failed += check(tensor_mul(&x, &y, &z) == 0, "mul through a strided view");
        failed += check(z.data[0] == 1.0f && z.data[1] == 4.0f && z.data[2] == 9.0f
                        && z.data[3] == 16.0f && z.data[4] == 25.0f && z.data[5] == 36.0f,
                        "mul squares each element");

        failed += check(tensor_create(&sum_axis0, along_row, 1) == 0, "create sum along axis 0");
        failed += check(tensor_create(&sum_axis1, along_col, 1) == 0, "create sum along axis 1");
        failed += check(tensor_create(&bad, bad_shape, 1) == 0, "create a wrong sum shape");
        tensor_fill(&bad, -1.0f);
        failed += check(tensor_sum(&x, 0, &bad) != 0, "sum rejects the wrong result shape");
        failed += check(bad.data[0] == -1.0f, "rejected sum leaves the destination unchanged");
        failed += check(tensor_sum(&x, -1, &sum_axis0) != 0, "sum rejects a negative axis");
        failed += check(tensor_sum(&x, 2, &sum_axis0) != 0, "sum rejects an axis past the rank");
        failed += check(tensor_sum(&x, 0, &sum_axis0) == 0, "sum axis 0");
        failed += check(sum_axis0.data[0] == 5.0f && sum_axis0.data[1] == 7.0f
                        && sum_axis0.data[2] == 9.0f,
                        "axis 0 sum is [5,7,9]");
        failed += check(tensor_sum(&y, 1, &sum_axis1) == 0, "sum axis 1 through a view");
        failed += check(sum_axis1.data[0] == 6.0f && sum_axis1.data[1] == 15.0f,
                        "axis 1 sum is [6,15]");

        tensor_destroy(&x);
        tensor_destroy(&y);
        tensor_destroy(&z);
        tensor_destroy(&sum_axis0);
        tensor_destroy(&sum_axis1);
        tensor_destroy(&bad);
    }

    tensor_destroy(&copied);
    tensor_destroy(&a);

    if (failed) {
        printf("test_tensor: FAIL (%d assertions broken)\n", failed);
        return 1;
    }
    printf("test_tensor: PASS\n\n");
    return 0;
}
