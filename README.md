# hindi-llm — a small Hindi language model, from scratch

Train a **~50M-parameter Hindi causal language model from scratch in PyTorch**
(nanoGPT-style), then fine-tune it into a minimal Hindi **chat** model — with a
readable, end-to-end pipeline you can defend decision-by-decision.

No `HuggingFace Trainer`, no `nn.Transformer`, no prebuilt model classes. Every
layer (RoPE, RMSNorm, SwiGLU, causal attention) and the entire training loop are
written by hand with shape comments throughout. The whole thing runs end-to-end
on the tiny tracked sample in seconds, and scales to a real single-GPU run.

> **Status / honesty note.** This repo ships the *pipeline*, not a trained model.
> No loss curves or perplexities are claimed until you run real training — those
> spots are marked `TODO`. The included `data/sample_hindi.txt` is far too small
> for a usable model; it exists only for smoke tests.

---

## Table of contents

- [Motivation](#motivation) · [Why Hindi](#why-hindi)
- [Pipeline overview](#end-to-end-pipeline)
- [Quickstart](#quickstart) · [Run each stage](#running-each-stage)
- [Data](#1-data-pipeline) · [Tokenizer](#2-tokenizer) · [Model](#3-model-architecture)
- [Training](#4-pretraining) · [SFT](#5-supervised-fine-tuning-sft) · [Eval & demo](#6-evaluation--demo)
- [Hardware](#expected-hardware) · [Limitations](#honest-limitations) · [Scaling 10×](#what-changes-at-10×)
- [Repo layout](#repository-layout) · [Docs](#documentation)

---

## Motivation

Most "train a GPT from scratch" projects use English and a single script. The aim
here is different: a **complete, auditable training-infra project** for a
**low-resource language**, where the unglamorous parts — data cleaning, dedup,
tokenizer fertility, loss masking, checkpointing/resume, honest evaluation —
matter as much as the model. Every choice is documented in [`docs/`](docs/) so it
can be defended in an interview.

## Why Hindi

Hindi is spoken by hundreds of millions of people yet is **under-served** by
LM tooling compared to English:

- English-centric tokenizers are highly *fertile* on Devanagari (many tokens per
  word), which inflates sequence length and training cost — so a **Hindi-specific
  tokenizer** is a real, measurable win (see [tokenizer notes](docs/tokenizer_notes.md)).
- Hindi web corpora are **noisier** and need careful cleaning/dedup; for a small
  model with little spare capacity, **data quality dominates**.
- It exercises real low-resource-NLP problems (script normalization, language
  filtering, mixed-script contamination) that English glosses over.

## End-to-end pipeline

```
   raw Hindi text (.txt / .jsonl / dir)
            │
            ▼
 [1] prepare_corpus.py   NFC normalize · boilerplate strip · quality filters
            │            · exact dedup (SHA-1) · near-dedup (MinHash+LSH)
            ▼            → clean.jsonl, clean.txt, corpus_stats.json, data_card.md
 [2] train_tokenizer.py  NFC → Metaspace → BPE  (32k, 7 special tokens)
            │            → hindi_bpe.json + fertility/compression/unknown report
            ▼
 [3] encode_dataset.py   tokenize + <eos> per doc → flat uint16 token stream
            │            → train.bin / val.bin / meta.json
            ▼
 [4] train.py            from-scratch GPT (RoPE+RMSNorm+SwiGLU), bf16, grad accum,
            │            cosine LR + warmup, clip, val ppl, resume, samples, wandb
            ▼            → checkpoints/base/{last,best}.pt
 [5] sft.py              chat template, assistant-only loss → chat model
            │            → checkpoints/sft/{last,best}.pt
            ▼
 [6] evaluate.py / generate.py / launch_gradio.py   perplexity · samples · demo
```

## Quickstart

With [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                      # create the env from pyproject
uv run pytest                # 19 fast CPU tests (model, tokenizer, mask, overfit, chat)

uv run python scripts/prepare_corpus.py  --help
uv run python scripts/train_tokenizer.py --help
uv run python scripts/encode_dataset.py  --help
uv run python scripts/train.py           --help
uv run python scripts/sft.py             --help
uv run python scripts/evaluate.py        --help
uv run python scripts/launch_gradio.py   --help
```

Without `uv` (pip fallback):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",demo,logging" for gradio + wandb
pytest
python scripts/train.py --help
```

End-to-end smoke run on the tracked sample (CPU, ~seconds — produces gibberish,
proves the wiring):

```bash
python scripts/prepare_corpus.py  --input data/sample_hindi.txt --output-dir data/processed --txt-mode line --min-chars 80
python scripts/train_tokenizer.py --input data/processed/clean.txt --output tokenizer/hindi_bpe.json --vocab-size 2000 --min-frequency 1
python scripts/encode_dataset.py  --input data/processed/clean.jsonl --tokenizer tokenizer/hindi_bpe.json --val-fraction 0.1
python scripts/train.py --config configs/hindi_50m.yaml \
  --set model.context_length=64 --set model.d_model=128 --set model.n_layers=2 --set model.n_heads=4 \
  --set train.batch_size=8 --set train.grad_accum_steps=1 --set train.max_steps=40 \
  --set train.eval_interval=20 --set train.sample_interval=0 --set checkpoint.out_dir=checkpoints/base
python scripts/generate.py --checkpoint checkpoints/base/best.pt --prompt "भारत एक" --max-new-tokens 30
```

## Running each stage

The config lives in [`configs/hindi_50m.yaml`](configs/hindi_50m.yaml) on top of
the dataclass defaults in [`src/hindi_llm/config.py`](src/hindi_llm/config.py).
Any field is overridable on the CLI with repeatable `--set dotted.key=value`.

### 1. Data pipeline

`scripts/prepare_corpus.py` accepts `.txt`, `.jsonl` (configurable text field),
or a directory. It NFC-normalizes Devanagari, strips boilerplate lines, applies
ordered quality filters (min length, Devanagari ratio, Latin/digit/punctuation
ratios, repeated-char runs, URL-heavy), removes **exact** duplicates (SHA-1) and
**near** duplicates (hand-rolled **MinHash + LSH** over character shingles), and
writes `clean.jsonl`, `clean.txt`, `corpus_stats.json`, and a generated
`data_card.md` (per-filter removal counts, length percentiles, Devanagari-ratio
summary, sample snippets). For a low-resource model this stage matters most:
dedup curbs memorization, language filtering curbs English contamination.
→ [`docs/data_card_template.md`](docs/data_card_template.md)

```bash
python scripts/prepare_corpus.py --input data/raw --output-dir data/processed \
  --min-chars 200 --min-devanagari 0.7
```

### 2. Tokenizer

`scripts/train_tokenizer.py` trains a Hindi **BPE** with the `tokenizers` library
directly (NFC → Metaspace → BPE) and 7 special tokens (`<pad> <bos> <eos> <unk>
<system> <user> <assistant>`). It writes a report with **fertility**
(tokens/word), **compression** (chars/token), **unknown rate**, examples, and a
baseline comparison (external tokenizer if supplied, else clearly-labeled
byte/char fallbacks). Vocab size (~32k) trades sequence length against the
embedding's share of a 50M budget. → [`docs/tokenizer_notes.md`](docs/tokenizer_notes.md)

```bash
python scripts/train_tokenizer.py --input data/processed/clean.txt \
  --output tokenizer/hindi_bpe.json --vocab-size 32000
```

### 3. Model architecture

A decoder-only GPT in [`src/hindi_llm/model.py`](src/hindi_llm/model.py),
built from scratch with shape comments on every nontrivial op:

- token embedding (tied to the output head) · **RoPE** rotary positions
- **RMSNorm** (pre-norm) · multi-head **causal** attention (fused SDPA *and* an
  explicit masked path) · **SwiGLU** MLP · residual connections

Default ≈ **48.5M params** (≈32.1M non-embedding): `d_model=512`, `n_layers=10`,
`n_heads=8` (head_dim 64), `context=1024`, `vocab=32000`. Print the breakdown:

```bash
python -m hindi_llm.config
```

→ [`docs/architecture_choices.md`](docs/architecture_choices.md)

### 4. Pretraining

`scripts/train.py` is a hand-written loop: **bf16** autocast on capable GPUs
(fp32 fallback), gradient accumulation, AdamW (decay on matrices only), **cosine
LR with linear warmup**, gradient clipping, periodic **validation perplexity**,
`last`/`best` checkpointing with atomic writes and **resume**, periodic Hindi
sample generations, an always-on JSONL metrics log, optional wandb, and tokens/s.
→ [`docs/training_notes.md`](docs/training_notes.md)

```bash
python scripts/train.py --config configs/hindi_50m.yaml          # full run
python scripts/train.py --config configs/hindi_50m.yaml --resume # continue
python scripts/train.py --config configs/hindi_50m.yaml --wandb  # + W&B logging
```

**Resume** restores model + optimizer + scaler + step + best-val and continues
the LR schedule from `checkpoints/base/last.pt`.

### 5. Supervised fine-tuning (SFT)

`scripts/sft.py` loads the base checkpoint (rebuilding the architecture from its
saved config), and fine-tunes on chat data using the template in
[`src/hindi_llm/chat_template.py`](src/hindi_llm/chat_template.py):

```
<system>
आप एक सहायक, स्पष्ट और ईमानदार हिंदी सहायक हैं।
</system>
<user>
{user_message}
</user>
<assistant>
{assistant_message}
</assistant>
```

Loss is **masked to assistant tokens only** (and the closing `<eos>`), base and
SFT checkpoints are kept **separate**, and three JSONL schemas are accepted
(`{system,user,assistant}`, `{instruction,input,output}`, `{messages:[...]}`).

```bash
python scripts/sft.py --config configs/hindi_50m.yaml \
  --set sft.data_path=data/your_sft.jsonl \
  --set sft.base_checkpoint=checkpoints/base/best.pt
```

### 6. Evaluation & demo

```bash
# perplexity + qualitative report (base and/or SFT, whichever exist)
python scripts/evaluate.py --config configs/hindi_50m.yaml

# one-off generation
python scripts/generate.py --checkpoint checkpoints/base/best.pt --prompt "भारत एक ऐसा देश है"
python scripts/generate.py --checkpoint checkpoints/sft/best.pt  --chat --prompt "भारत की राजधानी क्या है?"

# minimal Gradio Hindi chat demo (clearly labeled a research toy)
python scripts/launch_gradio.py --checkpoint checkpoints/sft/best.pt
```

`evaluate.py` writes `outputs/eval_report.md` with **real** validation perplexity,
Hindi generations, and explicit failure-mode notes. → [`docs/eval_notes.md`](docs/eval_notes.md)

## Results

Run on your own corpus, then fill in:

- `TODO: insert real loss curve after training` (from `outputs/metrics.jsonl`)
- `TODO: insert validation perplexity` (from `scripts/evaluate.py`)
- `TODO: insert qualitative Hindi samples` (from `outputs/eval_report.md`)
- `TODO: insert tokenizer fertility / compression` (from the tokenizer report)

## Expected hardware

- **Real small run**: a single **A100 (40/80GB)** or **RTX 4090 (24GB)**, bf16.
  The default config (1024 context, eff. batch 256 seqs, 20k steps ≈ 5B tokens
  seen) is sized to finish a meaningful run in **under 24h** on such a card.
  Lower `train.batch_size` / raise `grad_accum_steps` if you hit OOM.
- **Smoke tests / development**: CPU or Apple Silicon (MPS) — seconds, via the
  tiny config overrides shown above.

## Honest limitations

- A **~50M** model on a modest Hindi corpus is a research toy: expect repetition,
  occasional English code-switching, factual errors/hallucination, weak stopping,
  and Devanagari spelling slips on rare conjuncts.
- The tracked sample is **not** training data — bring your own corpus.
- Evaluation is perplexity + qualitative generations only; **no** standardized
  benchmark scores are claimed.
- No safety tuning. The Gradio app says, in Hindi and English, that it is a tiny
  research demo, not a production assistant.

## What changes at 10×

Most of it is config + infra, not a rewrite: a larger/cleaner corpus, a larger
vocab (lower fertility), wider/deeper model, longer context, larger batches, and
distributed training (DDP → FSDP/ZeRO, activation checkpointing). Full write-up:
→ [`docs/scaling_10x.md`](docs/scaling_10x.md)

## Repository layout

```text
configs/hindi_50m.yaml      reference run config (overrides on dataclass defaults)
scripts/                    prepare_corpus · train_tokenizer · encode_dataset
                            train · sft · generate · evaluate · launch_gradio
src/hindi_llm/              config · data · tokenizer_io · model · train_utils
                            sampling · chat_template · eval_utils
tests/                      tokenizer smoke · model shapes · causal mask
                            tiny overfit · chat template   (19 tests, CPU, fast)
docs/                       architecture · tokenizer · data card · training
                            eval · scaling notes
data/                       raw/interim/processed (gitignored) + tiny sample
```

## Documentation

- [Architecture choices](docs/architecture_choices.md)
- [Tokenizer notes](docs/tokenizer_notes.md)
- [Data card template](docs/data_card_template.md)
- [Training notes](docs/training_notes.md)
- [Evaluation notes](docs/eval_notes.md)
- [Scaling 10×](docs/scaling_10x.md)

## License

[MIT](LICENSE) © Abhinabha Barat.
