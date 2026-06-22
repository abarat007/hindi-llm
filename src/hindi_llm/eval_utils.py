"""Evaluation helpers: load a model from a checkpoint, compute corpus
perplexity deterministically, and produce qualitative generations.

Shared by scripts/evaluate.py, scripts/generate.py and scripts/launch_gradio.py
so "load a checkpoint" and "measure perplexity" mean the same thing everywhere.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .chat_template import Turn, build_chat_ids
from .config import ModelConfig
from .data import BinDataset
from .model import GPT, build_model
from .sampling import generate
from .tokenizer_io import HindiTokenizer
from . import train_utils as tu


def load_model_from_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Rebuild the model from the architecture saved in the checkpoint, then
    load the weights. Returns (model in eval mode, checkpoint dict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first (scripts/train.py)."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def corpus_perplexity(
    model: GPT,
    dataset: BinDataset,
    block_size: int,
    device: str,
    amp_dtype: torch.dtype | None = None,
    batch_size: int = 8,
    max_windows: int | None = None,
) -> dict[str, float]:
    """Deterministic perplexity over non-overlapping windows of the dataset.

    Unlike the random-batch training estimate, this sweeps the data in order so
    the number is reproducible. Returns {"loss", "ppl", "tokens"}.
    """
    data = dataset.data
    n = len(data)
    block_size = min(block_size, model.cfg.context_length)
    starts = list(range(0, n - block_size - 1, block_size))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("dataset too small for the requested block_size")

    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        x = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in chunk
        ]).to(device)                                  # [B, T]
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in chunk
        ]).to(device)                                  # [B, T]
        with tu.autocast_ctx(device, amp_dtype):
            _, loss = model(x, y)                       # mean NLL over B*T tokens
        ntok = x.numel()
        total_nll += loss.item() * ntok
        total_tokens += ntok

    mean = total_nll / total_tokens
    return {"loss": mean, "ppl": math.exp(min(mean, 20)), "tokens": total_tokens}


@torch.no_grad()
def complete_prompts(
    model: GPT,
    tok: HindiTokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Free-form completions from a *base* model (no chat template)."""
    gen = tu.make_generator(device, seed)
    out = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p, add_bos=True)], device=device)
        full = generate(model, ids, max_new_tokens, temperature=temperature,
                        top_k=top_k, top_p=top_p, eos_id=tok.eos_id, generator=gen)
        text = tok.decode(full[0].tolist())
        out.append({"prompt": p, "completion": text})
    return out


@torch.no_grad()
def chat_responses(
    model: GPT,
    tok: HindiTokenizer,
    user_prompts: list[str],
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Chat-formatted responses from an *SFT* model (assistant span only)."""
    gen = tu.make_generator(device, seed)
    out = []
    for up in user_prompts:
        ids, _ = build_chat_ids(tok, [Turn("user", up)], add_generation_prompt=True)
        prompt = torch.tensor([ids], device=device)
        full = generate(model, prompt, max_new_tokens, temperature=temperature,
                        top_k=top_k, top_p=top_p, eos_id=tok.eos_id, generator=gen)
        resp = tok.decode(full[0, len(ids):].tolist()).strip()
        out.append({"user": up, "assistant": resp})
    return out
