/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/ffnn/ffnn.h>
#include <serialcore/ffnn/dense.h>
#include <serialcore/ffnn/gemm.h>
#include <serialcore/iobin.h>
#include <serialcore/iojson.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ffnn_layer_ops_t *ops_for(layer_type_t type)
{
    if (type == FFNN_DENSE) return dense_layer_ops();
    return NULL;
}

static int layer_ready(const ffnn_layer_t *l)
{
    if (!l || !l->ops || !l->ops->forward || !l->ops->backward) return 0;
    if (!l->ops->output || !l->ops->delta || !l->ops->count_params) return 0;
    if (l->ops->shape && l->ops->shape(l) != 0) return 0;
    return 1;
}

ffnn_network_t *ffnn_create(int inputs, int batch, float learning_rate, float momentum, float decay)
{
    ffnn_network_t *net;

    if (inputs <= 0 || batch <= 0) return NULL;
    net = (ffnn_network_t *)calloc(1, sizeof(*net));
    if (!net) return NULL;

    net->cap = 8;
    net->layers = (ffnn_layer_t *)calloc((size_t)net->cap, sizeof(ffnn_layer_t));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    net->inputs = inputs;
    net->outputs = inputs;
    net->batch = batch;
    net->learning_rate = learning_rate;
    net->momentum = momentum;
    net->decay = decay;
    net->loss = ffnn_loss_mse;
    net->input_buffer = (float *)calloc((size_t)batch * (size_t)inputs, sizeof(float));
    net->input = net->input_buffer;
    if (!net->input_buffer) {
        ffnn_destroy(net);
        return NULL;
    }
    return net;
}

void ffnn_destroy(ffnn_network_t *net)
{
    if (!net) return;
    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        if (l->ops && l->ops->release) l->ops->release(l);
    }
    free(net->layers);
    free(net->input_buffer);
    mmpool_destroy(net->pool);
    free(net);
}

void ffnn_set_loss(ffnn_network_t *net, ffnn_loss_fn fn, void *userdata)
{
    if (!net) return;
    net->loss = fn ? fn : ffnn_loss_mse;
    net->loss_data = fn ? userdata : NULL;
}

void ffnn_loss_mse(const float *output, const float *target, float *delta, int n, void *userdata)
{
    (void)userdata;
    if (!output || !target || !delta || n <= 0) return;
    for (int i = 0; i < n; ++i) delta[i] = target[i] - output[i];
}

void ffnn_sgd_step(float *param, float *upd, int n, float lr, float momentum, float decay, int batch, int weight_decay)
{
    float scale;
    if (!param || !upd || n <= 0 || batch <= 0) return;
    scale = lr / (float)batch;
    for (int i = 0; i < n; ++i) {
        float g = upd[i];
        if (weight_decay) g -= decay * (float)batch * param[i];
        param[i] += scale * g;
        upd[i] *= momentum;
    }
}

int ffnn_add_layer(ffnn_network_t *net, int inputs, int outputs, layer_type_t type, activaton_t activation)
{
    const ffnn_layer_ops_t *ops;
    int expect;
    ffnn_layer_t *slot;

    if (!net || net->compiled) return -1;
    if (inputs <= 0 || outputs <= 0) return -1;

    ops = ops_for(type);
    if (!ops) {
        fprintf(stderr, "ffnn: layer type %d has no implementation\n", (int)type);
        return -1;
    }

    expect = (net->n == 0) ? net->inputs : net->layers[net->n - 1].outputs;
    if (inputs != expect) {
        fprintf(stderr, "ffnn: layer %d inputs=%d != expected %d\n", net->n, inputs, expect);
        return -1;
    }

    if (net->n == net->cap) {
        int new_cap = net->cap * 2;
        ffnn_layer_t *p = (ffnn_layer_t *)realloc(net->layers, (size_t)new_cap * sizeof(ffnn_layer_t));
        if (!p) return -1;
        net->layers = p;
        net->cap = new_cap;
    }

    slot = &net->layers[net->n];
    *slot = (ffnn_layer_t){0};
    slot->ops = ops;
    slot->type = type;
    slot->activation = activation;
    slot->inputs = inputs;
    slot->outputs = outputs;
    slot->batch = net->batch;

    if (type == FFNN_DENSE && dense_prepare(slot) != 0) {
        if (slot->ops && slot->ops->release) slot->ops->release(slot);
        *slot = (ffnn_layer_t){0};
        return -1;
    }

    net->n++;
    net->outputs = outputs;
    return 0;
}

int ffnn_compile(ffnn_network_t *net)
{
    int *counts;

    if (!net || net->n <= 0) return -1;
    if (net->compiled) return 0;

    counts = (int *)calloc((size_t)net->n, sizeof(int));
    if (!counts) return -1;

    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        if (!layer_ready(l)) {
            free(counts);
            return -1;
        }
        counts[i] = l->ops->count_params(l);
        if (counts[i] < 0) {
            free(counts);
            return -1;
        }
    }

    net->pool = mmpool_create(counts, net->n);
    free(counts);
    if (!net->pool) return -1;

    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        float *params = mmpool_params_at(net->pool, i);
        float *grads = mmpool_grads_at(net->pool, i);
        if (net->pool->offs[i] == net->pool->offs[i + 1]) continue;
        if (!l->ops->bind || l->ops->bind(l, params, grads) != 0) {
            mmpool_destroy(net->pool);
            net->pool = NULL;
            return -1;
        }
        if (l->ops->init) l->ops->init(l);
    }

    net->compiled = 1;
    return 0;
}

int ffnn_forward(ffnn_network_t *net, const float *input, float *output)
{
    ffnn_layer_t *last;
    float *y;
    int nout;

    if (!net || !input || !output) return -1;
    if (!net->compiled || !net->pool) {
        fprintf(stderr, "ffnn_forward: network not compiled — call ffnn_compile()\n");
        return -1;
    }

    net->input = net->input_buffer;
    for (int i = 0; i < net->batch * net->inputs; ++i) net->input[i] = input[i];

    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        float *dy = l->ops->delta(l);
        net->index = i;
        if (dy) gemm_fill(l->batch * l->outputs, 0.0f, dy, 1);
        if (l->ops->forward(l, net) != 0) return -1;
        net->input = l->ops->output(l);
        if (!net->input) return -1;
    }

    last = &net->layers[net->n - 1];
    y = last->ops->output(last);
    nout = net->batch * net->outputs;
    if (!y) return -1;
    for (int i = 0; i < nout; ++i) output[i] = y[i];
    return 0;
}

int ffnn_backward(ffnn_network_t *net, const float *target)
{
    ffnn_layer_t *last;
    float *y;
    float *dy;
    int nout;

    if (!net || !target || !net->compiled || !net->loss) return -1;

    last = &net->layers[net->n - 1];
    y = last->ops->output(last);
    dy = last->ops->delta(last);
    nout = net->batch * net->outputs;
    if (!y || !dy) return -1;
    net->loss(y, target, dy, nout, net->loss_data);

    for (int i = net->n - 1; i >= 0; --i) {
        ffnn_layer_t *l = &net->layers[i];
        ffnn_layer_t *prev = (i == 0) ? NULL : &net->layers[i - 1];

        if (prev) {
            net->input = prev->ops->output(prev);
            net->delta = prev->ops->delta(prev);
            if (net->delta) gemm_fill(prev->batch * prev->outputs, 0.0f, net->delta, 1);
        } else {
            net->delta = NULL;
        }
        net->index = i;
        if (l->ops->backward(l, net) != 0) return -1;
    }
    return 0;
}

int ffnn_update(ffnn_network_t *net)
{
    if (!net || !net->compiled) return -1;
    for (int i = 0; i < net->n; ++i) {
        ffnn_layer_t *l = &net->layers[i];
        if (l->ops->update) {
            l->ops->update(l, net->learning_rate, net->momentum, net->decay, net->batch);
        }
    }
    return 0;
}

int ffnn_train_step(ffnn_network_t *net, const float *input, const float *target)
{
    float *out;
    int saved;
    int rc;

    if (!net || !input || !target) return -1;
    out = (float *)malloc((size_t)net->batch * (size_t)net->outputs * sizeof(float));
    if (!out) return -1;

    saved = net->train;
    net->train = 1;
    rc = ffnn_forward(net, input, out);
    if (rc == 0) rc = ffnn_backward(net, target);
    if (rc == 0) rc = ffnn_update(net);
    net->train = saved;
    free(out);
    return rc;
}

int ffnn_predict(ffnn_network_t *net, const float *input, float *output)
{
    int saved;
    int rc;
    if (!net) return -1;
    saved = net->train;
    net->train = 0;
    rc = ffnn_forward(net, input, output);
    net->train = saved;
    return rc;
}

int ffnn_save(const ffnn_network_t *net, const char *json_path, const char *bin_path)
{
    cJSON *root;
    cJSON *layers;
    int rc;

    if (!net || !net->compiled || !net->pool || !json_path || !bin_path) return -1;
    if (net->pool->param_count > 0 && !net->pool->params) return -1;

    root = cJSON_CreateObject();
    if (!root) return -1;

    cJSON_AddStringToObject(root, "kind", nn_kind_str[NN_KIND_FFNN]);
    cJSON_AddNumberToObject(root, "inputs", net->inputs);
    cJSON_AddNumberToObject(root, "outputs", net->outputs);
    cJSON_AddNumberToObject(root, "batch", net->batch);
    cJSON_AddNumberToObject(root, "learning_rate", net->learning_rate);
    cJSON_AddNumberToObject(root, "momentum", net->momentum);
    cJSON_AddNumberToObject(root, "decay", net->decay);
    cJSON_AddNumberToObject(root, "param_count", net->pool->param_count);

    layers = cJSON_CreateArray();
    if (!layers) {
        cJSON_Delete(root);
        return -1;
    }
    cJSON_AddItemToObject(root, "layers", layers);

    for (int i = 0; i < net->n; i++) {
        const ffnn_layer_t *l = &net->layers[i];
        cJSON *layer = cJSON_CreateObject();
        cJSON *extra = NULL;

        if (!layer) {
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddStringToObject(layer, "type", layer_type_str[l->type]);
        cJSON_AddStringToObject(layer, "activation", activaton_name(l->activation));
        cJSON_AddNumberToObject(layer, "inputs", l->inputs);
        cJSON_AddNumberToObject(layer, "outputs", l->outputs);
        if (l->ops && l->ops->save_extra) extra = l->ops->save_extra(l);
        if (extra) cJSON_AddItemToObject(layer, "extra", extra);
        cJSON_AddItemToArray(layers, layer);
    }

    rc = iojson_write_cjson(json_path, root);
    cJSON_Delete(root);
    if (rc != 0) return -1;
    return iobin_write_floats(bin_path, net->pool->params, net->pool->param_count);
}

static int parse_layer_type(const char *s, layer_type_t *out)
{
    if (!s || !out) return -1;
    for (int t = 0; t <= (int)FFNN_SOFTMAX; t++) {
        if (strcmp(layer_type_str[t], s) == 0) {
            *out = (layer_type_t)t;
            return 0;
        }
    }
    return -1;
}

ffnn_network_t *ffnn_load(const char *json_path, const char *bin_path)
{
    cJSON *root;
    int inputs = 0, outputs = 0, batch = 0;
    float lr = 0.0f, momentum = 0.0f, decay = 0.0f;
    cJSON *layers;
    int n_layers;
    ffnn_network_t *net = NULL;
    int ok = 1;

    if (!json_path || !bin_path) return NULL;
    root = iojson_parse_file(json_path);
    if (!root || !cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return NULL;
    }

    if (iojson_get_int(root, "inputs", &inputs) != 0 ||
        iojson_get_int(root, "outputs", &outputs) != 0 ||
        iojson_get_int(root, "batch", &batch) != 0 ||
        iojson_get_float(root, "learning_rate", &lr) != 0 ||
        iojson_get_float(root, "momentum", &momentum) != 0 ||
        iojson_get_float(root, "decay", &decay) != 0) {
        cJSON_Delete(root);
        return NULL;
    }

    layers = iojson_get_array(root, "layers");
    n_layers = layers ? cJSON_GetArraySize(layers) : 0;
    if (n_layers <= 0) {
        cJSON_Delete(root);
        return NULL;
    }

    net = ffnn_create(inputs, batch, lr, momentum, decay);
    if (!net) {
        cJSON_Delete(root);
        return NULL;
    }

    for (int i = 0; ok && i < n_layers; i++) {
        cJSON *layer = cJSON_GetArrayItem(layers, i);
        const char *type_s = NULL;
        const char *act_s = NULL;
        int lin = 0, lout = 0;
        layer_type_t type = FFNN_BLANK;
        activaton_t act;
        cJSON *extra;

        if (!cJSON_IsObject(layer) ||
            iojson_get_string(layer, "type", &type_s) != 0 ||
            iojson_get_string(layer, "activation", &act_s) != 0 ||
            iojson_get_int(layer, "inputs", &lin) != 0 ||
            iojson_get_int(layer, "outputs", &lout) != 0 ||
            parse_layer_type(type_s, &type) != 0) {
            ok = 0;
            break;
        }
        act = activaton_type((char *)act_s);
        if (strcmp(activaton_name(act), act_s) != 0) {
            ok = 0;
            break;
        }
        if (ffnn_add_layer(net, lin, lout, type, act) != 0) {
            ok = 0;
            break;
        }
        extra = iojson_get_object(layer, "extra");
        if (net->layers[net->n - 1].ops->load_extra &&
            net->layers[net->n - 1].ops->load_extra(&net->layers[net->n - 1], extra) != 0) {
            ok = 0;
        }
    }

    if (ok && net->outputs != outputs) ok = 0;
    if (ok && ffnn_compile(net) != 0) ok = 0;
    if (ok && iobin_read_floats(bin_path, net->pool->params, net->pool->param_count) != 0) ok = 0;

    cJSON_Delete(root);
    if (!ok) {
        ffnn_destroy(net);
        return NULL;
    }
    return net;
}
