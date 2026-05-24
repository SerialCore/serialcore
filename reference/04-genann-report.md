# Genann Report

## Repository
- Path: `reference/genann`
- Upstream: `https://github.com/codeplea/genann.git`

## Purpose
Genann is a minimal C99 neural-network library for dense feedforward networks with backpropagation. Its design goal is simplicity: a tiny embeddable API, a single main implementation file, and straightforward serialization.

Primary files:
- `reference/genann/genann.h`
- `reference/genann/genann.c`
- `reference/genann/README.md`

## Functional Scope
The public API supports:
- create a network with `genann_init()`
- run inference with `genann_run()`
- train with `genann_train()`
- clone with `genann_copy()`
- serialize with `genann_read()` and `genann_write()`

The examples and tests demonstrate XOR, Iris classification, and manual weight manipulation.

## Main Architecture
The central type is `struct genann` in `reference/genann/genann.h`. It contains:
- topology values: inputs, hidden layers, hidden units, outputs
- activation function pointers
- total counts of weights and neurons
- flat buffers for weights, outputs, and deltas

One important design choice is contiguous allocation: `genann_init()` allocates the struct and all numeric arrays in a single block, then points into it. This keeps the library compact and cache-friendly.

## Algorithm Implementations

### Forward Pass
`reference/genann/genann.c` implements `genann_run()` as a simple loop over neurons and weights:
- input values are copied into the beginning of the output buffer
- each neuron computes a weighted sum plus bias
- the configured activation function is applied

Biases are represented as the first weight for each neuron and multiplied by `-1.0`.

### Training
`genann_train()` performs one online backprop update:
1. run a forward pass
2. compute output deltas
3. backpropagate hidden deltas
4. update output-layer weights
5. update hidden-layer weights

The implementation is compact and works well for the default sigmoid hidden/output behavior.

### Activation Handling
The library exposes several activations:
- sigmoid
- cached sigmoid
- threshold
- linear

The cached sigmoid path uses a lookup table for speed.

## Examples and Tests
Examples:
- `example1.c`: XOR via backprop
- `example2.c`: XOR via direct weight search
- `example3.c`: load and run saved model
- `example4.c`: Iris classifier

Tests in `reference/genann/test.c` cover:
- forward behavior
- simple learning cases
- serialization
- copy behavior
- sigmoid-cache correctness

These tests make the repo stronger than most minimal ANN examples.

## Serialization
`genann_write()` writes a plain-text format consisting of:
- topology header
- all weights

`genann_read()` reconstructs a network from the same format. This is easy to inspect and debug.

## Strengths
- Very small and easy to embed
- Clear contiguous-memory layout
- Examples and tests are well aligned with the API
- Good didactic implementation of simple dense networks

## Limitations and Quirks
- Serialization does not preserve activation-function selections; loading recreates default activations from `genann_init()`.
- Backprop derivatives are effectively coded for sigmoid hidden layers and sigmoid/linear outputs, so arbitrary activation-function pointers are not generally trainable.
- `genann_run()` returns a pointer into internal mutable memory.
- The API uses `const genann *` in places even though internal scratch buffers are mutated.
- There is no batching, optimizer support, regularization, softmax, or dataset abstraction.

## Overall Assessment
Genann is a strong reference for how to build a very small neural-network library in C. It is intentionally limited, but within that scope it is clean, practical, and much better structured than most toy examples.
