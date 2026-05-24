# GPT-2-in-C Report

## Repository
- Path: `reference/GPT-2-in-C`
- Upstream: `https://github.com/angry-kratos/GPT-2-in-C.git`

## Purpose
This repository is a small educational C implementation of a GPT-2-small-style transformer. It focuses on clarity and low-level implementation rather than production performance. The code is intended to show the mechanics of forward propagation and partial training logic in plain C.

Main files:
- `reference/GPT-2-in-C/gpt.c`
- `reference/GPT-2-in-C/timed.c`
- `reference/GPT-2-in-C/README.md`

## Functional Scope
- Load GPT-2-small weights from a `safetensors` file
- Run forward inference over token sequences
- Decode output tokens using a custom binary decoder file
- Provide a timing-instrumented variant in `timed.c`
- Contain a partial training/backward path for experimentation

The implementation is narrow and hard-coded to GPT-2 small dimensions.

## Main Architecture
`reference/GPT-2-in-C/gpt.c` defines several major structs:
- `TokenDecoder`
- `ModelParameters`
- `Gradients`
- `Activations`
- `BackwardActivations`

An important design choice is that model parameters are not copied into a separate structure. Instead, the code locates tensor offsets in the loaded `safetensors` blob and points struct fields directly into that payload.

The model dimensions are fixed in macros:
- vocab = 50257
- context = 1024
- embedding size = 768
- 12 heads
- 12 layers

## Algorithm Implementations

### Forward Pass
The main forward implementation is `process_transformer()` in `reference/GPT-2-in-C/gpt.c`. It performs:
1. token and positional embedding addition
2. repeated transformer blocks:
   - layer norm
   - combined QKV projection
   - causal self-attention
   - attention output projection
   - residual connection
   - second layer norm
   - MLP projection
   - GELU
   - MLP output projection
   - residual connection
3. final layer norm
4. output projection through tied embeddings
5. softmax over the full vocabulary

Attention is implemented with explicit nested loops over heads, query positions, key positions, and value channels. It is intentionally naive.

### Sampling
Generation uses greedy decoding only. The code selects the argmax token from the last-position softmax and appends it repeatedly. There is no temperature, top-k, or top-p sampling.

### Training Support
The code includes a partial backward path that computes output-layer cross-entropy gradients and some embedding gradients. It is not a full end-to-end transformer trainer, and there is no optimizer/update stage.

## Data and Tokenization Flow
The repo expects external assets:
- `model.safetensors`
- `data`
- `enc`

Important limitation: the C code does not implement text encoding. Input must already be tokenized into the binary data format that the program expects. Output decoding is handled via the `enc` file.

## Benchmarking
`reference/GPT-2-in-C/timed.c` adds ad hoc timing around:
- model loading
- GEMM-heavy stages
- softmax
- overall inference loop

This file is mainly for profiling the baseline implementation.

## Strengths
- Clear low-level transformer forward pass in C
- Explicit activation and gradient storage
- Useful educational baseline for GPT mechanics
- Timing variant helps identify hotspots

## Limitations and Quirks
- Hard-coded to GPT-2 small shapes
- `safetensors` parsing is done with brittle string matching, not a real parser
- Training support is incomplete
- Mode selection depends on compile-time macro choices rather than a clean CLI
- Greedy-only generation
- Very memory-heavy because it stores full activations and gradient buffers
- README data-format instructions do not appear fully aligned with what the C loader expects

## Overall Assessment
This repo is a useful educational walkthrough of GPT-2 forward propagation in C, but it is not a robust inference or training system. Its value is in showing how the transformer is laid out and executed with plain arrays and loops.
