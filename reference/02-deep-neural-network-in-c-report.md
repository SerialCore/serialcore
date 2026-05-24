# Deep-Neural-Network-in-C Report

## Repository
- Path: `reference/Deep-Neural-Network-in-C`
- Upstream: `https://github.com/mounirouadi/Deep-Neural-Network-in-C.git`

## Purpose
This repository is a compact educational implementation of a feedforward neural network in C for MNIST digit classification. The entire system is effectively centered in a single file, `reference/Deep-Neural-Network-in-C/main.c`.

## Functional Scope
- Load MNIST-like binary image/label files
- Train a fixed-topology MLP
- Save model weights to `model.bin`
- Run test-set predictions after training

There is no CLI argument handling, no architecture configuration, and no separate inference mode exposed in `main()`.

## Main Architecture
The model is a hard-coded one-hidden-layer MLP:
- input: 784 features
- hidden: 256 units
- output: 10 units

Core definitions are at the top of `reference/Deep-Neural-Network-in-C/main.c`.

The code uses large global arrays for:
- full training and test datasets
- network weights and biases
- metrics counters

This makes the implementation easy to inspect but rigid and memory-heavy.

## Algorithm Implementations

### Forward Pass
Forward propagation is implemented directly in loops inside:
- `train()` in `reference/Deep-Neural-Network-in-C/main.c`
- `test()` in `reference/Deep-Neural-Network-in-C/main.c`

The math is:
- hidden preactivation = input * `weight1` + `bias1`
- hidden activation = sigmoid
- output preactivation = hidden * `weight2` + `bias2`
- output activation = sigmoid
- prediction = argmax of output vector

### Backpropagation
Training logic is also in `train()`. It computes:
- output error as `target - output`
- propagated terms through the output and hidden layers
- direct SGD-like updates to `weight1`, `weight2`, `bias1`, `bias2`

However, the indexing and gradient logic for the first layer appears incorrect. The code uses `delta1[i]` when updating `weight1[i][j]`, even though the hidden-layer gradient should align with the hidden neuron index `j`. This is a major correctness concern.

## Data and Serialization

### Dataset Loading
Dataset loading is implemented in `load_mnist()` in `reference/Deep-Neural-Network-in-C/main.c`.

The loader reads raw bytes from:
- `mnist_train_images.bin`
- `mnist_train_labels.bin`
- `mnist_test_images.bin`
- `mnist_test_labels.bin`

It normalizes pixel bytes to `[0, 1]` and one-hot encodes labels.

Important issue: the loader does not parse or skip standard IDX headers. The included file sizes strongly suggest the files do contain IDX headers, so the loader likely shifts all data by the header bytes and reads corrupted examples.

### Model Save/Load
Serialization is implemented by:
- `save_weights_biases()`
- `load_weights_biases()`

The code writes raw native `double` arrays in a fixed order with no metadata or versioning. This is simple, but only safe when the architecture and platform match exactly.

## Strengths
- Extremely easy to read
- Demonstrates the full MLP workflow in one file
- Useful as a teaching example for basic forward propagation, sigmoid activations, one-hot labels, and weight persistence

## Limitations and Correctness Concerns
- Likely misreads MNIST due to not skipping IDX headers
- Metric reporting is broken because `main()` shadows global counters with local variables
- Backpropagation for the first layer appears wrong
- Uses sigmoid outputs for multiclass classification instead of softmax + cross-entropy
- No batching, shuffling, validation split, or configurable topology
- README compile instructions likely need `-lm` on Linux because of `exp()`

## Overall Assessment
This repo is best treated as an educational prototype rather than a correct MNIST trainer. It clearly demonstrates the intended structure of an MLP implementation in C, but the dataset parsing and gradient-update issues mean it should not be used as a trusted reference for algorithm correctness without fixing those bugs first.
