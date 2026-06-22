"""Autoregressive sampling: temperature, top-k, and top-p (nucleus).

A single :func:`generate` drives token-by-token decoding. Filtering is applied to
the logits of the *last* position before sampling. Greedy decoding is the
temperature -> 0 limit and is selected explicitly when ``temperature == 0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest logits per row; set the rest to -inf.

    logits: [B, V] -> [B, V]
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    # kth largest value per row -> [B, 1]
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass >= p; mask the rest.

    logits: [B, V] -> [B, V]
    """
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)  # [B, V]
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)                       # [B, V]
    # mask tokens once cumulative prob has exceeded p (shift so the first token
    # crossing the threshold is kept)
    remove = cumprobs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # scatter back to the original vocab order
    return sorted_logits.gather(-1, sorted_idx.argsort(dim=-1))


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Continue ``idx`` for up to ``max_new_tokens`` tokens.

    idx: [B, T_prompt] -> returns [B, T_prompt + n_generated]
    Stops early once every row has emitted ``eos_id`` (if provided).
    """
    was_training = model.training
    model.eval()
    device = idx.device
    context_length = model.cfg.context_length
    B = idx.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        # never feed more than the model's context window
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        logits, _ = model(idx_cond)                       # [B, t, V]
        logits = logits[:, -1, :]                          # [B, V] last step

        if temperature == 0.0:
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1] greedy
        else:
            logits = logits / temperature
            if top_k:
                logits = top_k_filter(logits, top_k)
            if top_p:
                logits = top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)              # [B, V]
            # multinomial requires the generator to live on the same device as
            # probs; if a mismatched (e.g. CPU) generator was passed, fall back
            # to the global RNG (still reproducible when the seed was set).
            if generator is not None and generator.device.type == probs.device.type:
                next_tok = torch.multinomial(probs, 1, generator=generator)  # [B, 1]
            else:
                next_tok = torch.multinomial(probs, 1)     # [B, 1]

        if eos_id is not None:
            # rows already finished keep emitting eos (stay benign)
            next_tok = torch.where(finished.unsqueeze(1),
                                   torch.full_like(next_tok, eos_id), next_tok)
            finished = finished | (next_tok.squeeze(1) == eos_id)

        idx = torch.cat([idx, next_tok], dim=1)            # [B, T+1]
        if eos_id is not None and bool(finished.all()):
            break

    if was_training:
        model.train()
    return idx
