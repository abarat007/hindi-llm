# Scaling 10×

What changes if you take this from ~50M params / (say) ~1B tokens to **~500M
params** and/or **10× the data**. The code is written so most of this is config,
not rewrites.

## 10× parameters (~50M → ~500M)

Roughly: `params ∝ n_layers × d_model²`. Reasonable ~500M shapes:

| | 50M (this repo) | ~500M target |
|---|---|---|
| `d_model` | 512 | ~1280 |
| `n_layers` | 10 | ~24 |
| `n_heads` | 8 | ~20 (head_dim 64) |
| `context_length` | 1024 | 2048 |
| vocab | 32k | 48–64k (now affordable) |

Consequences:

- **Memory**: activations and optimizer state grow with params. AdamW keeps 2
  fp32 moments per weight (~8 bytes/param) + master weights. ~500M no longer fits
  comfortably with a naive setup on one 24GB card during training — see
  distributed/memory below.
- **Embedding share shrinks**: at `d_model=1280`, a 64k embedding is ~82M of
  ~500M (~16%), vs ~34% here. That's why a **larger vocab becomes worth it** at
  scale (lower fertility, shorter sequences) — revisit the tokenizer.
- **Initialization/stability**: the GPT-2 scaled residual init already accounts
  for depth; deeper models may want slightly lower peak LR and longer warmup.

## 10× data

- **Don't just repeat data.** More *unique* cleaned Hindi tokens is the point;
  repeating the same corpus 10× mostly teaches memorization. The dedup stage
  ([`prepare_corpus.py`](../scripts/prepare_corpus.py)) matters more, not less, at
  scale.
- **Chinchilla rule of thumb**: ~20 tokens/param is compute-optimal. ~500M params
  → ~10B tokens. So 10× params *and* ~10× data tend to go together.
- **Streaming/sharding**: a 10B-token `uint16` shard is ~20GB. Keep `np.memmap`
  (already used) and consider multiple shard files + a sampler over them. The
  encoder may need to write incrementally rather than buffering all ids in RAM.
- **Quality filtering pays off more**: at 10B tokens, a 1% boilerplate/English
  contamination is 100M wasted tokens. Tighten the filters and re-inspect.

## Batch size

- Larger models like **larger batches** (more stable gradients). Scale the
  *effective* batch via `grad_accum_steps` and, when you have multiple GPUs, data
  parallelism. Watch for the large-batch generalization gap; scale LR with batch
  (roughly linear, with warmup) and keep gradient clipping on.
- Token throughput, not step count, is the budget. Track tokens/sec and total
  tokens seen.

## Context length

- 1024 → 2048 (or more). Attention is `O(T²)` in time and memory, so longer
  context is expensive. RoPE already supports longer sequences and can be
  extended by raising `rope_theta` or applying RoPE scaling for lengths beyond
  training. Flash attention (the SDPA path) is essential at long context.

## Tokenizer reconsideration

- Retrain the BPE on the larger, cleaner corpus with a **larger vocab** (48–64k):
  lower Hindi fertility, shorter sequences, better compute efficiency, and the
  embedding is now a smaller share of the budget.
- Re-check fertility/compression and the unknown rate on held-out domains.

## Distributed training

- **DDP** (Distributed Data Parallel) is the first step: replicate the model per
  GPU, all-reduce gradients. The training loop would wrap the model in DDP and
  shard the data sampler; the core loop logic is unchanged.
- For models that don't fit per GPU: **FSDP** / ZeRO sharding of params,
  gradients, and optimizer state; activation checkpointing to trade compute for
  memory; mixed precision throughout.
- `torch.compile` (already a config flag) gives a meaningful speedup at scale.

## Evaluation improvements

- Per-domain held-out perplexity (news vs conversational vs technical).
- A small **human-rated** Hindi instruction benchmark for the SFT model.
- Standardized Indic benchmarks (e.g. IndicGLUE-style tasks) once the model is
  strong enough to register signal.
- Safety/refusal probes and bias checks before any wider release.

## What stays the same

The architecture (RoPE + RMSNorm + SwiGLU decoder), the from-scratch training
loop, the data/tokenizer/eval pipeline, and the config-driven design all carry
over. Scaling is mostly **config + infrastructure** (distributed, sharding,
throughput), not a redesign — which is the point of keeping the code readable and
parameterized.
