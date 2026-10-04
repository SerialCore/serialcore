/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_TYPES
#define SERIALCORE_TYPES

/* Neural-network model kind. */
typedef enum nn_kind {
    NN_KIND_FFNN,
    NN_KIND_SONN
} nn_kind_t;

extern const char *nn_kind_str[];

/* FFNN layer types. */
typedef enum layer_type {
    FFNN_BLANK,
    FFNN_DENSE,
    FFNN_CONVOLUTIONAL,
    FFNN_MAXPOOL,
    FFNN_BATCHNORM,
    FFNN_SOFTMAX
} layer_type_t;

extern const char *layer_type_str[];

/* SONN algorithm types. */
typedef enum sonn_type {
    GNG,
    SOM,
    NGAS
} sonn_type_t;

extern const char *sonn_type_str[];

#endif
