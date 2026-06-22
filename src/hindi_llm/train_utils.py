"""Shared training utilities: seeding, device/precision, LR schedule,
loss/perplexity estimation, and checkpoint I/O.

Kept separate from the train script so SFT and evaluation reuse the exact same
helpers (one definition of "how we save a checkpoint", "how the LR is computed",
etc.).
"""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Determinism + device + precision
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device: str, seed: int) -> "torch.Generator | None":
    """A seeded RNG on the given device for reproducible sampling.

    Also seeds the global RNG so that, if the device-specific generator cannot be
    created (some backends), sampling still falls back to a seeded global RNG.
    """
    torch.manual_seed(seed)
    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    except (RuntimeError, TypeError):
        return None


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_amp_dtype(pref: str, device: str) -> torch.dtype | None:
    """Pick the autocast dtype. Returns None for full fp32.

    Policy (per the project's training notes):
      * "auto": bf16 on bf16-capable CUDA, else fp32. (No silent fp16: fp16
        needs loss scaling and is less stable; opt in explicitly.)
      * explicit "bfloat16"/"float16"/"float32" are honored.
    Autocast is only used on CUDA here; MPS/CPU run fp32 for predictability.
    """
    if device != "cuda":
        return None
    if pref == "float32":
        return None
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float16":
        return torch.float16
    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None  # fall back to fp32 rather than unscaled fp16


def autocast_ctx(device: str, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


# --------------------------------------------------------------------------- #
# Learning-rate schedule: linear warmup -> cosine decay -> floor
# --------------------------------------------------------------------------- #
def get_lr(
    step: int,
    warmup_steps: int,
    lr_decay_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    # 2) past the decay horizon -> constant floor
    if step >= lr_decay_steps:
        return min_lr
    # 3) cosine decay from max_lr down to min_lr
    progress = (step - warmup_steps) / max(lr_decay_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


# --------------------------------------------------------------------------- #
# Loss / perplexity estimation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(
    model,
    datasets: dict,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, dict[str, float]]:
    """Average loss over ``eval_iters`` batches for each split; also report
    perplexity = exp(loss)."""
    was_training = model.training
    model.eval()
    out: dict[str, dict[str, float]] = {}
    for split, ds in datasets.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = ds.get_batch(batch_size, block_size, device)
            with autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)
            losses[i] = loss.item()
        mean = losses.mean().item()
        out[split] = {"loss": mean, "ppl": math.exp(min(mean, 20))}
    if was_training:
        model.train()
    return out


def grad_global_norm(model) -> float:
    """L2 norm of all gradients (for logging; computed after backward)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    step: int,
    best_val: float,
    config: dict[str, Any],
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": config,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    # write to a temp file then rename, so an interrupted save can't corrupt a
    # good checkpoint
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    device: str = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False because our checkpoints intentionally store the
    # optimizer state and a config dict, not just model tensors. Only load
    # checkpoints you trust/produced yourself.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# --------------------------------------------------------------------------- #
# Lightweight JSONL metrics logger (always on, independent of wandb)
# --------------------------------------------------------------------------- #
class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
