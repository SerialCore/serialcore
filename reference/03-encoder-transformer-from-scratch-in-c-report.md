# Encoder-Transformer-From-Scratch-in-C Report

## Repository
- Path: `reference/Encoder-Transformer-From-Scratch-in-C`
- Upstream: `https://github.com/KartikeyBartwal/Encoder-Transformer-From-Scratch-in-C.git`

## Purpose
This repository is an educational attempt to build a Transformer-like text model in C. It contains both a `Pseudo Version` and a `Final Version`, but even the final code is still prototype-grade and better understood as a learning project than a correct Transformer implementation.

Primary files:
- `reference/Encoder-Transformer-From-Scratch-in-C/README.md`
- `reference/Encoder-Transformer-From-Scratch-in-C/Final Version/main.c`

## Functional Scope
The final pipeline tries to perform:
- text loading and sentence splitting
- tokenization and vocabulary building
- word embedding construction
- positional encoding
- self-attention-like processing
- a feed-forward prediction stage
- MSE-based weight updates

It aims at next-token-style learning, but the actual implementation predicts a small embedding vector, not a token distribution.

## Main Architecture
The executable path is driven by `reference/Encoder-Transformer-From-Scratch-in-C/Final Version/main.c`, which orchestrates modules for:
- text loading and cleaning
- tokenization
- preprocessing
- transformer block logic
- feed-forward layer logic
- backpropagation helpers

Key modules:
- `Data_Loading_Cleaning.c`
- `Tokenizer.c`
- `Data_Preprocessing.c`
- `transformer_block.c`
- `feed_forward_layer.c`
- `backpropagation.c`

Several files, including `self_attention_layer.c` and `test.c`, are standalone prototypes rather than part of a clean library structure.

## Algorithm Implementations

### Tokenization and Embeddings
`reference/Encoder-Transformer-From-Scratch-in-C/Final Version/Tokenizer.c` implements a hash-table-backed word-to-token mapping. Token IDs begin at 1, and 0 acts as padding or empty space.

Each word also receives a tiny 2D embedding. These are not initialized from a learned distribution in a standard way; they are generated from random factors combined with `sin(token_id)` and `cos(token_id)`. This is more of a pedagogical embedding sketch than a standard trainable embedding table.

### Positional Encoding
`reference/Encoder-Transformer-From-Scratch-in-C/Final Version/Data_Preprocessing.c` adds positional encoding through small sinusoidal offsets. The dimensions are fixed and tiny, matching the hard-coded `EMBEDDING_DIM=2` style design in `main.c`.

### Self-Attention
Attention-like logic lives in `reference/Encoder-Transformer-From-Scratch-in-C/Final Version/transformer_block.c`. The code loads fixed-size query/key/value weights and computes matrix products, but it does not implement standard Transformer attention of the form:

`softmax(QK^T / sqrt(d_k)) V`

over full sequence positions. It is better described as a simplified matrix-processing analogue to self-attention.

### Feed-Forward and Loss
The output stage in `main.c` and `feed_forward_layer.c` applies small learned matrices and non-linearities such as `leaky_relu` and `swish` to produce a 2D predicted embedding. Loss is computed as MSE against the embedding of the held-out final token.

### Backpropagation
`reference/Encoder-Transformer-From-Scratch-in-C/Final Version/backpropagation.c` does not implement full chain-rule-based Transformer backpropagation. Instead, it uses heuristic update rules derived from loss and current weights/activations. This makes the code useful for intuition-building, but not as a mathematically correct training reference.

## Training and Inference Flow
The main training-like loop:
1. Read text from `text_data.txt`
2. Split into sentences
3. Build token IDs and padded token matrices
4. Remove the last real token from each sentence as the target
5. Convert remaining tokens to embeddings
6. Apply positional encoding and simplified attention
7. Predict the missing token's embedding
8. Compute MSE
9. Update weights

There is no proper inference or decoding path:
- no softmax over vocabulary
- no nearest-neighbor token decoder from predicted embedding
- no autoregressive generation loop

## Strengths
- Shows the intended structure of a Transformer pipeline in C
- Separates text loading, tokenization, preprocessing, attention, and backprop into modules
- Useful as a learning scaffold for someone exploring Transformer components manually

## Limitations and Quirks
- Not a mathematically correct Transformer implementation
- Tiny fixed dimensions make it more toy than general
- Uses hard-coded absolute file paths for some weights in `main.c`
- Only a subset of sentences are cleaned consistently
- Header/source constants are inconsistent in places
- Lacks a real token-probability objective and usable generation path

## Overall Assessment
This repo is a teaching prototype for Transformer ideas, not a reliable implementation of the algorithm. It demonstrates the pipeline concepts and module boundaries, but the attention math, training objective, and backpropagation are too simplified to serve as a solid reference for correct Transformer training.
