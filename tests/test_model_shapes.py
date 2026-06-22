"""Shape / contract tests for the GPT model.

These run on CPU with a tiny config in well under a second.
"""

from __future__ import annotations

import math

import torch

from hindi_llm.config import ModelConfig, estimate_num_params
from hindi_llm.model import build_model


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=256, context_length=32, d_model=64, n_layers=2, n_heads=4)
    base.update(kw)
    return ModelConfig(**base)


def test_model_initializes():
    m = build_model(tiny_cfg())
    assert sum(p.numel() for p in m.parameters()) > 0


def test_forward_logits_shape():
    cfg = tiny_cfg()
    m = build_model(cfg)
    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = m(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None


def test_loss_with_targets():
    cfg = tiny_cfg()
    m = build_model(cfg)
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    tgt = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = m(idx, tgt)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0  # scalar
    # at init, loss should sit near ln(vocab) for a uniform-ish distribution
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0


def test_param_count_matches_estimator():
    cfg = tiny_cfg()
    m = build_model(cfg)
    assert m.num_params() == estimate_num_params(cfg)["total"]


def test_tied_vs_untied_embeddings():
    tied = build_model(tiny_cfg(tie_embeddings=True))
    untied = build_model(tiny_cfg(tie_embeddings=False))
    # tying shares the [V, D] table -> fewer total params
    assert untied.num_params() > tied.num_params()
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()


def test_sequence_too_long_raises():
    cfg = tiny_cfg(context_length=8)
    m = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 9))
    try:
        m(idx)
    except ValueError:
        return
    raise AssertionError("expected ValueError for T > context_length")
