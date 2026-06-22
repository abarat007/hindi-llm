# Architecture choices

This document explains *why* the model is built the way it is, so every decision
is defensible. The model lives in [`src/hindi_llm/model.py`](../src/hindi_llm/model.py)
and is configured in [`src/hindi_llm/config.py`](../src/hindi_llm/config.py).

## The model at a glance

A decoder-only (GPT-style) transformer:

```
tokens ─► embedding ─► [ RMSNorm ─► RoPE attention ─► +residual
                        RMSNorm ─► SwiGLU MLP      ─► +residual ] × N
       ─► final RMSNorm ─► (tied) linear head ─► logits
```

Default (`configs/hindi_50m.yaml`): `d_model=512`, `n_layers=10`, `n_heads=8`
(`head_dim=64`), `context_length=1024`, `vocab=32000`, SwiGLU hidden `1408`,
weights tied → **~48.5M parameters** (~32.1M non-embedding). Run
`python -m hindi_llm.config` to print the exact breakdown.

## Attention (multi-head causal self-attention)

- **Causal**: position *t* may only attend to positions ≤ *t*. Enforced either
  by an explicit lower-triangular mask (the readable `manual` path) or by
  `F.scaled_dot_product_attention(is_causal=True)` (the fast `sdpa` path, which
  uses flash-attention kernels on GPU). Both produce identical outputs (verified
  in tests to ~1e-6).
- **Multi-head**: `H=8` heads of dim 64. Multiple low-dimensional heads let the
  model attend to several relationships in parallel; `head_dim=64` is a sweet
  spot for flash kernels and numerical stability.
- **No biases** in the projections (LLaMA-style). Biases add parameters and
  empirically don't help at this scale.

## Positions: RoPE vs learned absolute embeddings

We use **Rotary Position Embeddings (RoPE)** rather than learned absolute
position embeddings.

| | Learned absolute | RoPE (chosen) |
|---|---|---|
| Params | extra `context × d_model` table | **none** |
| Relative position | not explicit | **falls out of the dot product** |
| Length extrapolation | poor beyond trained length | better (can extend `theta`) |

RoPE rotates query/key channel pairs by an angle proportional to position, so the
attention score between positions *i* and *j* depends on *i − j*. This gives
relative-position awareness for free and removes a parameter table — useful when
the embedding already dominates a 50M budget. Sinusoidal absolute embeddings are
the other zero-parameter option, but RoPE consistently trains better for LMs.

## Normalization: RMSNorm vs LayerNorm

We use **RMSNorm**. LayerNorm subtracts the mean and divides by the standard
deviation, then applies a learned scale *and* bias. RMSNorm drops the
mean-centering and the bias, normalizing only by the root-mean-square:

```
y = x / sqrt(mean(x²) + eps) * weight
```

It is cheaper (no mean, no bias), numerically simpler, and matches LayerNorm
quality for transformers (used by LLaMA, T5-style models). We compute it in
float32 even under bf16/fp16 for stability. Placement is **pre-norm** (norm
*before* each sub-layer), which keeps a clean residual highway and makes deep
stacks trainable without warmup gymnastics.

## Feed-forward: SwiGLU vs GELU/ReLU MLP

A standard MLP is `down(act(up(x)))` with hidden width `4·d_model`. **SwiGLU**
adds a multiplicative gate:

```
SwiGLU(x) = down( SiLU(gate(x)) ⊙ up(x) )
```

It has three matrices (`gate`, `up`, `down`) instead of two. To keep the
parameter count comparable to a `4·d` ReLU MLP, the hidden width is set to
`≈ (8/3)·d_model` (here 512 → 1408, rounded to a multiple of 128). The gate gives
a data-dependent, multiplicative path that improves quality per parameter; it is
the standard FFN in modern LLMs (PaLM, LLaMA).

## Tied vs untied embeddings

We **tie** the input embedding and the output projection (they share one
`vocab × d_model` matrix). At `vocab=32000, d_model=512` that table is ~16.4M
params — a third of the model. Untying would add another 16.4M for the head with
little quality gain at this scale, so tying is the efficient choice and is
standard for small models. Set `model.tie_embeddings: false` to compare.

## Depth / width / heads tradeoffs

- **Width (`d_model`)** drives the cost of every matmul (`O(d²)` per token in
  attention projections, `O(d·hidden)` in the MLP) and the embedding size.
- **Depth (`n_layers`)** adds representational power roughly linearly in params
  but increases the critical path (harder to parallelize, more latency).
- For ~50M params with a 32k vocab, a moderately deep, moderately wide model
  (10 × 512) balances capacity against the fixed embedding cost. Going much
  wider inflates the embedding share; going much deeper risks under-training each
  layer on a small corpus.
- **Heads**: `n_heads` must divide `d_model`; we keep `head_dim=64` for good
  kernel efficiency, so `n_heads = d_model/64`.

## Parameter-count reasoning

Per the estimator in `config.py` (which mirrors the modules exactly):

```
embedding         = vocab × d                     = 32000 × 512 = 16.38M
attention / layer = 4 × d²                        = 4 × 512²    = 1.05M
SwiGLU / layer    = 3 × d × hidden                = 3×512×1408  = 2.16M
norms / layer     = 2 × d                          (negligible)
per layer         ≈ 3.21M  →  × 10 layers          = 32.12M
lm_head           = 0 (tied)
TOTAL             ≈ 48.5M  (non-embedding ≈ 32.1M)
```

The non-embedding count (~32M) is the "real" model capacity; the embedding is a
lookup table whose size is set by the tokenizer's vocab. This is why tokenizer
vocab size is an architecture decision, not just a data decision — see
[`tokenizer_notes.md`](tokenizer_notes.md).
