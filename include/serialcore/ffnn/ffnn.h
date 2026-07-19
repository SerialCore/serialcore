/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_FFNN
#define SERIALCORE_FFNN_FFNN

#include <serialcore/sonn/activaton.h>
#include <serialcore/ffnn/mmpool.h>

/*
 * FFNN — feedforward, dense-matrix, fixed-topology neural network core.
 *
 * This is the canonical home for layered networks whose weights live in flat
 * matrices rather than per-neuron slots (the SONN core in src/sonn owns the
 * per-neuron + dynamic-adjacency model). Each layer family shipped here has
 * a matching source-file/header pair: dense.{c,h} for the fully-connected
 * layer, gemm.{c,h} for the GEMM + BLAS kernels. The runtime that threads
 * the layers together is declared here and implemented in src/ffnn/ffnn.c.
 */

typedef enum layertype {
    FFNN_BLANK = 0,
    FFNN_DENSE,
    FFNN_CONVOLUTIONAL,
    FFNN_MAXPOOL,
    FFNN_BATCHNORM,
    FFNN_SOFTMAX
} layertype_t;

struct ffnn_layer;
struct ffnn_network;
typedef struct ffnn_layer ffnn_layer_t;
typedef struct ffnn_network ffnn_network_t;
typedef void (*ffnn_forward_fn)(ffnn_layer_t *l, ffnn_network_t *net);
typedef void (*ffnn_backward_fn)(ffnn_layer_t *l, ffnn_network_t *net);
typedef void (*ffnn_update_fn)  (ffnn_layer_t *l, float lr, float momentum, float decay, int batch);

struct ffnn_layer {
    layertype_t type;
    activaton_t activation;

    int inputs;              /* the size of the 1-D vector the previous layer emits */
    int outputs;             /* the size this layer emits. */
    int batch;

    /* Parameters (host memory). Row-major: weights[o*inputs + i]. */
    float *weights;          /* [outputs * inputs] */
    float *biases;           /* [outputs] */
    float *weight_updates;   /* [outputs * inputs] */
    float *bias_updates;     /* [outputs] */

    float *output;           /* [batch * outputs] */
    float *pre_act;          /* [batch * outputs] — z before activation */
    float *delta;            /* [batch * outputs] */
    float *input_snapshot;   /* [batch * inputs] */

    /* Hook table. May be NULL for layers without learnable params (pool). */
    ffnn_forward_fn forward;
    ffnn_backward_fn backward;
    ffnn_update_fn update;
};

struct ffnn_network {
    int           n;              /* number of layers in `layers` */
    int           cap;            /* allocated slots in `layers` */
    ffnn_layer_t *layers;

    int           inputs;         /* network input size == layer[0].inputs */
    int           outputs;        /* network output size == layer[n-1].outputs */
    int           batch;          /* SGD batch size (1 for the unit tests) */

    float         learning_rate;
    float         momentum;
    float         decay;

    float        *input;          /* the input to the current layer during forward, and the output of the previous layer. */
    float        *input_buffer;   /* each forward resets `net->input` to `input_buffer` so the caller's data has a stable home to land in */
    float        *delta;

    int           train;          /* 1 → forward also caches inputs for backward */
    int           index;          /* current layer index during forward/backward */
    int           compiled;       /* 1 → ffnn_compile() has built net->pool and wired per-layer weight/bias views into it */

    /* Parameter memory pool. NULL until ffnn_compile() is called; once
     * built, every layer's `weights` / `biases` / `weight_updates` /
     * `bias_updates` points into pool->params / pool->grads. Serialization
     * is then a single fwrite/fread over pool->params. */
    mmpool_t     *pool;
};

/* Lifecycle */
ffnn_network_t* ffnn_create(int inputs, int batch, float learning_rate, float momentum, float decay);
void ffnn_destroy(ffnn_network_t *net);

/* Append a new layer to the network. 
 * The layer is built in place by the internal factory. Only the per-batch
 * workspace (output / pre_act / delta / input_snapshot) is allocated here;
 * the learnable parameters live in `net->pool`, which is built later by
 * ffnn_compile() from the collected layer shapes. Calling forward /
 * backward / update before ffnn_compile() is undefined. */
int ffnn_add_layer(ffnn_network_t *net, int inputs, int outputs, layertype_t type, activaton_t activation);

/* Build net->pool from the layers added so far and wire each layer's
 * weight / bias / weight_update / bias_update pointers into the pool's
 * contiguous arenas. He-init (formerly done in ffnn_build_layer) runs here,
 * via the pool's generate_random_parameter path. Must be called exactly
 * once after the last ffnn_add_layer; calling it again is a no-op. */
int ffnn_compile(ffnn_network_t *net);

/* Top-level passes. */
void ffnn_forward(ffnn_network_t *net, const float *input, float *output);

/* Backprop inverse-pass: computes per-layer deltas and accumulates weight/
 * bias updates. The output layer's delta is driven by (target - output).
 * Must be preceded by a forward pass on the same `net->input` buffer. */
void ffnn_backward(ffnn_network_t *net, const float *target);

/* Apply accumulated updates via SGD + momentum + (L2) decay. */
void ffnn_update(ffnn_network_t *net);

/* Convenience: forward + backward + update in one shot. */
void ffnn_train_step(ffnn_network_t *net, const float *input, const float *target);

/* Forward-only convenience; `output` must have net->outputs floats. */
void ffnn_predict(ffnn_network_t *net, const float *input, float *output);

#endif