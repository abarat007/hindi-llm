"""A GPT-style decoder-only transformer, built from scratch.

No ``nn.Transformer``, no prebuilt attention blocks. The architecture follows
the modern small-LM recipe (LLaMA-flavored):

  * token embedding (optionally tied to the output projection)
  * Rotary Position Embeddings (RoPE) instead of learned absolute positions
  * RMSNorm instead of LayerNorm, pre-norm placement
  * multi-head causal self-attention
  * SwiGLU feed-forward instead of GELU/ReLU MLP
  * residual connections around each sub-layer

Every nontrivial tensor op carries a shape comment. The notation used:
    B  = batch size
    T  = sequence length (time steps)
    D  = d_model (model width)
    H  = number of attention heads
    Dh = head dim = D // H
    V  = vocab size
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-subtraction, no bias).

    Cheaper than LayerNorm and empirically as good for LMs. We compute the
    normalization in float32 for numerical stability even under bf16/fp16.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        dtype = x.dtype
        x = x.float()
        # rms over the feature dim -> [B, T, 1]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms                                   # [B, T, D] normalized
        return (x.to(dtype)) * self.weight            # scale by learned gain


# --------------------------------------------------------------------------- #
# Rotary Position Embeddings (RoPE)
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE up to ``max_seq_len``.

    RoPE rotates pairs of query/key channels by an angle proportional to the
    token position. Relative position falls out of the dot product, so the model
    generalizes over positions without any learned position parameters.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        # inverse frequencies for each channel pair -> [Dh/2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()         # [T_max]
        freqs = torch.outer(t, inv_freq)              # [T_max, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [T_max, Dh]
        # buffers move with .to(device)/.half() but are not learned
        self.register_buffer("cos", emb.cos(), persistent=False)  # [T_max, Dh]
        self.register_buffer("sin", emb.sin(), persistent=False)  # [T_max, Dh]

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # return the leading seq_len rows -> [T, Dh] each
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # split the last dim in two halves and rotate: [..., Dh] -> [..., Dh]
    x1, x2 = x.chunk(2, dim=-1)                        # each [..., Dh/2]
    return torch.cat((-x2, x1), dim=-1)               # [..., Dh]


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k : [B, H, T, Dh]   cos, sin : [T, Dh]
    cos = cos[None, None, :, :]                        # [1, 1, T, Dh]
    sin = sin[None, None, :, :]                        # [1, 1, T, Dh]
    q_rot = q * cos + _rotate_half(q) * sin            # [B, H, T, Dh]
    k_rot = k * cos + _rotate_half(k) * sin            # [B, H, T, Dh]
    return q_rot, k_rot


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Two attention backends share the exact same math:
      * ``sdpa``   : torch's fused scaled_dot_product_attention (fast; flash
                     kernels on GPU). Causality via ``is_causal=True``.
      * ``manual`` : explicit scores -> causal mask -> softmax -> weighted sum,
                     written out so the mechanism (and the mask) is inspectable.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        # fused QKV projection: D -> 3D
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.attn_bias)
        # output projection: D -> D
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        impl = cfg.attn_impl
        if impl == "auto":
            impl = "sdpa" if hasattr(F, "scaled_dot_product_attention") else "manual"
        self.impl = impl

        # Precomputed lower-triangular causal mask for the manual path.
        # [1, 1, T_max, T_max] so it broadcasts over batch and heads.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.bool()[None, None], persistent=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape                              # [B, T, D]

        qkv = self.qkv(x)                              # [B, T, 3D]
        q, k, v = qkv.split(self.d_model, dim=2)       # each [B, T, D]

        # reshape into heads and move head dim forward
        # [B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)              # [B, H, T, Dh] each

        if self.impl == "sdpa":
            # fused, numerically stable, flash kernels when available
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )                                          # [B, H, T, Dh]
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            # attention scores: [B, H, T, T]
            att = (q @ k.transpose(-2, -1)) * scale
            # mask future positions (upper triangle) before softmax
            att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)               # [B, H, T, T]
            att = self.attn_dropout(att)
            y = att @ v                                # [B, H, T, Dh]

        # merge heads back: [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_dropout(self.proj(y))           # [B, T, D]
        return y


# --------------------------------------------------------------------------- #
# Feed-forward (SwiGLU)
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """SwiGLU MLP: down( SiLU(gate(x)) * up(x) ).

    The element-wise gate gives the FFN a multiplicative, data-dependent path
    that plain GELU/ReLU MLPs lack. Three matrices (gate, up, down); the hidden
    width is chosen ~8/3·D so the parameter count matches a 4·D ReLU MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.resolve_mlp_hidden()
        self.gate = nn.Linear(cfg.d_model, hidden, bias=cfg.mlp_bias)  # D -> Hd
        self.up = nn.Linear(cfg.d_model, hidden, bias=cfg.mlp_bias)    # D -> Hd
        self.down = nn.Linear(hidden, cfg.d_model, bias=cfg.mlp_bias)  # Hd -> D
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = F.silu(self.gate(x)) * self.up(x)          # [B, T, Hd] gated hidden
        return self.dropout(self.down(h))              # [B, T, D]


# --------------------------------------------------------------------------- #
# Transformer block (pre-norm)
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # pre-norm residual: x + sublayer(norm(x))
        x = x + self.attn(self.norm1(x), cos, sin)     # [B, T, D]
        x = x + self.mlp(self.norm2(x))                # [B, T, D]
        return x


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # [V, D]
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # D -> V

        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)

        if cfg.tie_embeddings:
            # weight tying: input embedding and output projection share weights
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init for residual output projections: keeps the variance
        # of the residual stream from growing with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=cfg.init_std * scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # subtract the token embedding table; if tied, lm_head shares it so
            # there is nothing extra to subtract.
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: [B, T] token ids, targets: [B, T] (or None)
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(
                f"sequence length {T} exceeds context_length "
                f"{self.cfg.context_length}"
            )

        x = self.tok_emb(idx)                          # [B, T, D]
        x = self.drop(x)
        cos, sin = self.rope(T)                        # [T, Dh] each
        # keep RoPE tables in the activation dtype (matters under autocast)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)                     # [B, T, D]
        x = self.norm_f(x)                             # [B, T, D]

        logits = self.lm_head(x)                       # [B, T, V]

        loss = None
        if targets is not None:
            # flatten time and batch for cross-entropy; ignore_index=-100 lets
            # SFT mask out prompt tokens by setting their target to -100.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),      # [B*T, V]
                targets.view(-1),                      # [B*T]
                ignore_index=-100,
            )
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on 2D params only (matrices/embeddings), not
        on 1D params (RMSNorm gains, biases). This is the standard LM recipe."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # fused AdamW is much faster on CUDA when available
        use_fused = "cuda" in str(next(self.parameters()).device)
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **extra)


def build_model(cfg: ModelConfig) -> GPT:
    """Convenience constructor used by scripts and tests."""
    return GPT(cfg)
