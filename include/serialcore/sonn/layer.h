/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_LAYER
#define SERIALCORE_SONN_LAYER

#include <serialcore/sonn/network.h>
#include <serialcore/sonn/activaton.h>

/*
 * Layer utilities for building a classic layered fully-connected
 * feedforward network as a *special implementation* on top of the SONN core.
 *
 * IMPORTANT:
 *   - The core network_t and nnpool know NOTHING about layers.
 *   - All layer metadata (ranges, activations) lives in an internal per-network
 *     table inside layer.c.
 *   - Connection weights use Option A (stored in receiver's neuron.weights[s]
 *     where s is the local edge slot).
 */

/* Build a feedforward topology.
 * layer_sizes[0] = input dim, layer_sizes[num_layers-1] = output dim.
 * layer_acts[i] is the activation for layer i (can be NULL for defaults).
 *
 * Neurons are created sequentially:
 *   layer 0 neurons: ids [0 .. layer_sizes[0]-1]
 *   layer 1 neurons: next block, etc.
 *
 * Full connections (directed) are added from layer i to layer i+1.
 * Connection weights are stored in the target neuron's weights[] vector
 * (indexed by local edge slot on receiver, Option A).
 */
int layer_build_ffnn(network_t *net, const int *layer_sizes, int num_layers, const activaton_t *layer_acts);

/* Forward pass for the layered FFNN.
 * 'inputs' must have length = size of first layer.
 * 'outputs' must have length = size of last layer.
 * Returns 0 on success.
 */
int layer_ffnn_forward(network_t *net, const float *inputs, float *outputs);

/* One supervised training step using backpropagation (full feedforward training).
 * Computes deltas for all layers and updates weights and biases throughout the network
 * using the chain rule and activation derivatives.
 *
 * target: desired output for the single output neuron (for XOR we use 1 unit output).
 * lr: learning rate.
 */
int layer_ffnn_train_step(network_t *net, const float *inputs, float target, float lr);

/* Helper: get the starting neuron id and count for a given layer index.
 * After layer_build_ffnn, layers are 0-based.
 */
int layer_get_range(network_t *net, int layer_idx, int *start_id, int *count);

#endif
