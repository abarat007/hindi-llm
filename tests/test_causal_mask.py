"""Causality tests.

A causal LM must not let position t attend to positions > t. We verify this two
ways: (1) the explicit manual-attention mask is lower-triangular, and (2) at the
model level, perturbing a *future* token leaves earlier positions' logits
unchanged — for both the manual and SDPA backends.
"""

from __future__ import annotations

import torch

from hindi_llm.config import ModelConfig
from hindi_llm.model import CausalSelfAttention, build_model


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=128, context_length=32, d_model=64, n_layers=2, n_heads=4)
    base.update(kw)
    return ModelConfig(**base)


def test_mask_is_lower_triangular():
    attn = CausalSelfAttention(tiny_cfg())
    T = 16
    mask = attn.causal_mask[0, 0, :T, :T]  # [T, T] bool
    expected = torch.tril(torch.ones(T, T)).bool()
    assert torch.equal(mask, expected)


def _future_does_not_leak(impl: str) -> None:
    cfg = tiny_cfg(attn_impl=impl)
    m = build_model(cfg)
    m.eval()
    B, T = 1, 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits_a, _ = m(idx)

    # change only the LAST token; positions < T-1 must be unaffected
    idx2 = idx.clone()
    cur = idx2[0, -1].item()
    idx2[0, -1] = (cur + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_b, _ = m(idx2)

    # earlier positions identical
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    # the changed (last) position should differ
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


def test_future_tokens_do_not_leak_manual():
    _future_does_not_leak("manual")


def test_future_tokens_do_not_leak_sdpa():
    _future_does_not_leak("sdpa")
