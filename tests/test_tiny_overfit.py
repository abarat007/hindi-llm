"""Tiny overfit test: the model must be able to memorize a single batch.

If a from-scratch transformer can't drive the loss on one fixed batch close to
zero, something is wrong with the forward/backward path (a classic sanity check
recommended by Karpathy). We train a small model on one random batch for a few
dozen steps and assert the loss collapses.
"""

from __future__ import annotations

import torch

from hindi_llm.config import ModelConfig
from hindi_llm.model import build_model


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, context_length=16, d_model=64, n_layers=2, n_heads=4,
        dropout=0.0,
    )
    model = build_model(cfg)
    model.train()

    B, T = 4, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    opt = model.configure_optimizers(weight_decay=0.0, lr=3e-3, betas=(0.9, 0.95))

    _, first_loss = model(idx, targets)
    first = first_loss.item()

    last = first
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        opt.step()
        last = loss.item()

    # the loss must drop dramatically — the model has memorized the batch
    assert last < 0.5, f"overfit failed: start={first:.3f} end={last:.3f}"
    assert last < first * 0.1
