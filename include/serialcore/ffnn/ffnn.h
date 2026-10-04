/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_FFNN_FFNN
#define SERIALCORE_FFNN_FFNN

#include <serialcore/math/activaton.h>
#include <serialcore/cJSON.h>
#include <serialcore/ffnn/mmpool.h>
#include <serialcore/types.h>

/* FFNN — sequential feedforward network. Each layer is an ops table plus
 * private state. The driver only chains output and delta along the stack. */

struct ffnn_layer;
struct ffnn_network;
typedef struct ffnn_layer ffnn_layer_t;
typedef struct ffnn_network ffnn_network_t;

typedef struct ffnn_layer_ops {
    layer_type_t type;
    int  (*shape)(const ffnn_layer_t *l);
    int  (*count_params)(const ffnn_layer_t *l);
    int  (*bind)(ffnn_layer_t *l, float *params, float *grads);
    void (*init)(ffnn_layer_t *l);
    int  (*forward)(ffnn_layer_t *l, ffnn_network_t *net);
    int  (*backward)(ffnn_layer_t *l, ffnn_network_t *net);
    void (*update)(ffnn_layer_t *l, float lr, float momentum, float decay, int batch);
    float *(*output)(ffnn_layer_t *l);
    float *(*delta)(ffnn_layer_t *l);
    cJSON *(*save_extra)(const ffnn_layer_t *l);
    int  (*load_extra)(ffnn_layer_t *l, const cJSON *extra);
    void (*release)(ffnn_layer_t *l);
} ffnn_layer_ops_t;

struct ffnn_layer {
    const ffnn_layer_ops_t *ops;
    void                   *state;

    layer_type_t type;
    activaton_t  activation;
    int          inputs;
    int          outputs;
    int          batch;
};

/* Writes the loss gradient into delta[0 .. n). MSE stores (target - output). */
typedef void (*ffnn_loss_fn)(const float *output, const float *target, float *delta, int n, void *userdata);

struct ffnn_network {
    int           n;
    int           cap;
    ffnn_layer_t *layers;

    int           inputs;
    int           outputs;
    int           batch;

    float         learning_rate;
    float         momentum;
    float         decay;

    float        *input;
    float        *input_buffer;
    float        *delta;

    int           train;          /* 1 → layers cache inputs for backward */
    int           index;
    int           compiled;

    ffnn_loss_fn  loss;
    void         *loss_data;

    mmpool_t     *pool;
};

ffnn_network_t *ffnn_create(int inputs, int batch, float learning_rate, float momentum, float decay);
void ffnn_destroy(ffnn_network_t *net);

/* Rejects types that have no ops table, and any call after ffnn_compile. */
int ffnn_add_layer(ffnn_network_t *net, int inputs, int outputs, layer_type_t type, activaton_t activation);

/* Build the parameter pool from each layer's count_params(), bind, then init.
 * A second call is a no-op. Fails if any layer has no forward. */
int ffnn_compile(ffnn_network_t *net);

/* NULL restores mean squared error. */
void ffnn_set_loss(ffnn_network_t *net, ffnn_loss_fn fn, void *userdata);
void ffnn_loss_mse(const float *output, const float *target, float *delta, int n, void *userdata);

/* Momentum SGD. weight_decay != 0 subtracts decay * param inside the step.
 * upd is then multiplied by momentum. */
void ffnn_sgd_step(float *param, float *upd, int n, float lr, float momentum, float decay, int batch, int weight_decay);

int ffnn_forward(ffnn_network_t *net, const float *input, float *output);
int ffnn_backward(ffnn_network_t *net, const float *target);
int ffnn_update(ffnn_network_t *net);
int ffnn_train_step(ffnn_network_t *net, const float *input, const float *target);
int ffnn_predict(ffnn_network_t *net, const float *input, float *output);

int ffnn_save(const ffnn_network_t *net, const char *json_path, const char *bin_path);
ffnn_network_t *ffnn_load(const char *json_path, const char *bin_path);

#endif
