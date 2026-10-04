/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/types.h>

const char *nn_kind_str[] = {
    "FFNN",
    "SONN"
};

const char *layer_type_str[] = {
    "BLANK",
    "DENSE",
    "CONVOLUTIONAL",
    "MAXPOOL",
    "BATCHNORM",
    "SOFTMAX"
};

const char *sonn_type_str[] = {
    "GNG",
    "SOM",
    "NGAS"
};
