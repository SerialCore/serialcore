# llama2.c Report

## Repository
- Path: `reference/llama2.c`
- Upstream: `https://github.com/karpathy/llama2.c.git`

## Purpose
`llama2.c` is a compact full-stack Llama-style project: train a small model in PyTorch, export it, and run inference in pure C. It prioritizes readability and end-to-end completeness over feature breadth.

Primary files:
- `reference/llama2.c/model.py`
- `reference/llama2.c/train.py`
- `reference/llama2.c/export.py`
- `reference/llama2.c/run.c`
- `reference/llama2.c/runq.c`

## Functional Scope
- Scratch training of small Llama-like models in Python
- Checkpoint export from repo/Hugging Face/Meta-compatible sources
- Float32 C inference in `run.c`
- Int8 quantized inference in `runq.c`
- Tokenizer export and TinyStories dataset tooling

## Main Architecture

### Python Model
`reference/llama2.c/model.py` implements a decoder-only Llama-style transformer with:
- RMSNorm
- RoPE
- SwiGLU FFN
- bias-free projections
- optional grouped-query attention through `n_kv_heads`
- tied input/output embeddings

### C Runtime
`reference/llama2.c/run.c` mirrors the model for single-token autoregressive inference. It maintains KV caches and applies:
- embedding lookup
- RMSNorm
- Q/K/V projections
- rotary embedding
- causal attention over cached tokens
- output projection
- SwiGLU FFN
- final norm and logits

The quantized runtime in `reference/llama2.c/runq.c` uses Q8_0-style int8 weights with per-group scales and dynamic activation quantization.

## Algorithm Implementations

### Attention and RoPE
Attention is implemented explicitly in C loops over heads and cached positions. RoPE is applied on the fly rather than loaded from disk. This keeps the checkpoint smaller and the runtime logic transparent.

### Sampling
`reference/llama2.c/run.c` supports:
- greedy decoding when `temperature == 0`
- multinomial sampling
- top-p sampling

This makes the C runtime more complete than many educational examples.

### Quantization
`reference/llama2.c/runq.c` performs int8 matrix multiplication with:
- quantized weights
- runtime-quantized activations
- int32 accumulation
- scale-based dequantization

Norms and some other operations remain in float.

## Training, Export, and Data Flow

### Training
`reference/llama2.c/train.py` supports:
- AdamW
- gradient accumulation
- cosine learning-rate decay with warmup
- mixed precision
- DDP
- checkpointing and evaluation

### Export
`reference/llama2.c/export.py` can:
- export repo checkpoints
- import/export Meta and Hugging Face formats
- write legacy fp32 and newer versioned formats
- write quantized checkpoints

### Tokenizer and Dataset
- `tokenizer.py` exports SentencePiece models to the repo's binary tokenizer format
- `tinystories.py` handles data download, tokenizer training, and pretokenized dataset generation

## Strengths
- Clean end-to-end training/export/inference story
- Readable PyTorch model and readable C runtime
- Includes a practical int8 inference path
- Good educational bridge between research code and systems code

## Limitations and Quirks
- The project is intentionally minimal and not production-hardened
- `run.c` and `runq.c` do not consume exactly the same checkpoint format generation path
- Chat mode in `run.c` is acknowledged as rough and lightly tested
- Python generation does not use a KV cache and is inefficient for long generation
- The C tokenizer format omits some metadata, and the code comments call this out
- Some import/export assumptions may not generalize cleanly across all HF grouped-query variants

## Overall Assessment
`llama2.c` is one of the stronger educational LLM codebases in this set. It provides a coherent path from training to export to fast-enough C inference, while still remaining compact enough to study directly.
