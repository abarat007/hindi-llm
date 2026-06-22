# Training notes

How the pretraining loop ([`scripts/train.py`](../scripts/train.py),
[`src/hindi_llm/train_utils.py`](../src/hindi_llm/train_utils.py)) works and how
to read what it prints.

## Mixed precision (bf16) with fp32 fallback

- On a bf16-capable GPU (A100, RTX 4090) the forward/backward run under
  `torch.autocast(dtype=bfloat16)`. bf16 has the same exponent range as fp32, so
  it needs **no loss scaling** — simpler and stable.
- If bf16 isn't available we fall back to **fp32** (not unscaled fp16). fp16 is
  supported only if you explicitly set `train.dtype: float16`, in which case a
  `GradScaler` is used. On CPU/MPS we run fp32 for predictability.
- Master weights stay fp32 in the optimizer; only the compute is reduced
  precision. This roughly halves memory and ~2× throughput on GPU.

## Gradient accumulation

The optimizer steps once per `grad_accum_steps` micro-batches. Effective batch =
`batch_size × grad_accum_steps` sequences; each loss is divided by
`grad_accum_steps` so the accumulated gradient equals the average over the large
batch. This lets a single 24GB/80GB GPU train with a large *effective* batch
without OOM. Tokens/optimizer-step = `batch_size × grad_accum × context_length`
(default: 32 × 8 × 1024 ≈ 262K tokens/step).

## AdamW

`betas=(0.9, 0.95)` (not 0.999 — the lower β2 reacts faster to the changing loss
landscape of LM pretraining). Weight decay (0.1) is applied **only to 2D tensors**
(matrices, embeddings), not to RMSNorm gains or biases — decaying norm scales
hurts. See `GPT.configure_optimizers`.

## LR schedule: linear warmup → cosine decay → floor

`train_utils.get_lr`:

1. **Warmup** (first `warmup_steps`): LR rises linearly from ~0 to the peak.
   Early gradients are large and noisy; ramping up avoids destabilizing the
   freshly-initialized model.
2. **Cosine decay**: LR follows a half-cosine from peak down to `min_lr` over the
   decay horizon. A smooth decay lets the model take big steps early and fine
   steps late, which generalizes better than a constant LR.
3. **Floor**: after the horizon, LR holds at `min_lr` (~10% of peak).

## Gradient clipping

Global-norm clip at `1.0`. If the total gradient norm exceeds the threshold, all
gradients are scaled down to it. This caps the occasional huge gradient (from a
bad batch or a sharp loss region) that would otherwise blow up the weights. The
logged `gnorm` is the *pre-clip* norm — watch it.

## What to watch in the logs

Each `log_interval`: `loss`, `lr`, `gnorm`, `tok/s`. Each `eval_interval`:
train/val loss and **val perplexity** (`exp(loss)`).

- **Healthy run**: loss falls fast then slowly; `gnorm` is stable (single digits,
  not spiking); val perplexity tracks down with train loss.
- **Divergence** (LR too high): loss suddenly shoots up or becomes `NaN`/`inf`;
  `gnorm` spikes hard. Fix: lower `optim.lr`, increase `warmup_steps`, confirm
  clipping is on. (Divergence early often means warmup is too short.)
- **Plateau too early**: loss flattens well above expectation. Could be LR too
  low (try higher peak), too little data (repeating → memorizing), or model too
  small. Check the train/val gap.
- **LR too low**: loss decreases but painfully slowly and `gnorm` stays tiny.
- **Overfitting**: train loss keeps dropping while val loss rises. At 50M on a
  modest corpus you're usually *under*-fitting, so this is more likely with a
  small corpus + many epochs — add data or dropout.

## Validation perplexity

Perplexity = `exp(cross-entropy)`. Intuition: the model's effective branching
factor — "on average it is as unsure as if choosing uniformly among PPL tokens."
Lower is better; it is comparable **only** for a fixed tokenizer/vocab (changing
the tokenizer changes the units). Use it to compare runs, not to compare against
models with different tokenizers.

## Checkpointing & resume

- `last.pt` (latest) and `best.pt` (lowest val loss) are saved separately. Writes
  are atomic (temp file + rename) so an interrupted save can't corrupt a good
  checkpoint.
- Checkpoints store model + optimizer + scaler + step + best_val + the full
  config. Resume with `--resume` (or `checkpoint.resume: true`): it restores all
  of these and continues the LR schedule from the saved step.
- A `config_snapshot.yaml` is written to the run directory so the exact run is
  reproducible.

## Logging

- A **local JSONL metrics log is always written** (`outputs/metrics.jsonl`),
  independent of any external service — one line per log/eval event.
- **wandb** is optional (`logging.wandb_enabled: true` or `--wandb`). If wandb
  isn't installed the run continues with a warning.

## Determinism

`set_seed` seeds Python/NumPy/Torch(+CUDA). Note: full bitwise determinism on GPU
also needs deterministic kernels; we seed for reproducibility of data order and
init, which is what matters for comparing runs.

## A tiny smoke run (CPU, seconds)

```bash
python scripts/train.py --config configs/hindi_50m.yaml \
  --set model.context_length=64 --set model.d_model=128 \
  --set model.n_layers=2 --set model.n_heads=4 \
  --set train.batch_size=8 --set train.grad_accum_steps=1 \
  --set train.max_steps=20 --set train.eval_interval=10 \
  --set train.sample_interval=0 --set train.device=cpu
```
