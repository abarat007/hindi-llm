"""Central configuration for the whole project.

Everything that can be tuned lives here, grouped into small dataclasses and
gathered under one root :class:`Config`. Scripts load a YAML (e.g.
``configs/hindi_50m.yaml``), which is merged on top of these defaults, so the
dataclasses are the single source of truth for *what* is configurable and the
YAML only records *overrides*.

Design choices:

- Plain ``dataclasses`` (no pydantic / hydra) to keep the magic at zero.
- Grouped sub-configs (data / tokenizer / model / ...) so each script touches
  only what it needs, while still being "one central config".
- A parameter-count estimator (:func:`estimate_num_params`) so we can defend the
  "~50M parameters" claim with arithmetic rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# Special tokens, in a fixed order. Index order is *not* relied upon anywhere;
# we always look tokens up by string. <pad> is first purely by convention.
SPECIAL_TOKENS: list[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<system>",
    "<user>",
    "<assistant>",
]


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where corpora live and how the train/val split is formed."""

    raw_dir: str = "data/raw"                 # source files you supply
    processed_dir: str = "data/processed"     # cleaned corpus + token shards
    cleaned_jsonl: str = "data/processed/clean.jsonl"
    cleaned_txt: str = "data/processed/clean.txt"
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    text_field: str = "text"                  # JSONL field holding the document text
    val_fraction: float = 0.0005              # fraction of tokens held out for val
    # For real corpora val_fraction ~0.0005 gives a few hundred K val tokens.
    # On the tiny sample we override this from the CLI so val isn't empty.


@dataclass
class TokenizerConfig:
    """Hindi BPE tokenizer settings (trained with the `tokenizers` library)."""

    path: str = "tokenizer/hindi_bpe.json"
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))
    # NFC Unicode normalization is applied; see tokenizer_io / train_tokenizer.


@dataclass
class ModelConfig:
    """GPT decoder dimensions. See docs/architecture_choices.md for the why."""

    vocab_size: int = 32000
    context_length: int = 1024
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8                 # head_dim = d_model / n_heads = 64 (clean)
    # SwiGLU hidden width. If None it is derived as
    #   round_to(mlp_ratio * d_model, mlp_multiple_of)
    # The 8/3 ratio keeps SwiGLU's 3 matrices ~param-equal to a 4x ReLU MLP.
    mlp_hidden: int | None = None
    mlp_ratio: float = 8 / 3
    mlp_multiple_of: int = 128
    dropout: float = 0.0             # 0.0 is fine while underfitting at this scale
    rope_theta: float = 10000.0      # RoPE base frequency
    rms_norm_eps: float = 1e-5
    attn_bias: bool = False          # LLaMA-style: no biases in linear layers
    mlp_bias: bool = False
    tie_embeddings: bool = True      # share input embedding with output projection
    init_std: float = 0.02           # GPT-2-style normal init std

    def resolve_mlp_hidden(self) -> int:
        """Return the concrete SwiGLU hidden width, deriving it if unset."""
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        raw = self.mlp_ratio * self.d_model
        m = self.mlp_multiple_of
        # round up to the nearest multiple of m
        return int(((int(raw) + m - 1) // m) * m)

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        return self.d_model // self.n_heads


@dataclass
class OptimConfig:
    """AdamW hyper-parameters + gradient clipping."""

    lr: float = 3e-4                 # peak LR (after warmup)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95              # 0.95 (not 0.999) is standard for LM pretrain
    eps: float = 1e-8
    grad_clip: float = 1.0           # max global grad norm; 0 disables clipping


@dataclass
class SchedulerConfig:
    """Cosine LR schedule with linear warmup. See train_utils.get_lr."""

    warmup_steps: int = 200
    # If lr_decay_steps is None it defaults to TrainConfig.max_steps.
    lr_decay_steps: int | None = None
    min_lr: float = 3e-5             # final LR floor (~10% of peak is typical)


@dataclass
class TrainConfig:
    """Runtime knobs for the pretraining loop."""

    batch_size: int = 32             # sequences per micro-step (per forward)
    grad_accum_steps: int = 8        # micro-steps per optimizer step
    max_steps: int = 20000           # optimizer steps (not micro-steps)
    eval_interval: int = 500
    eval_iters: int = 100            # batches averaged for train/val loss estimate
    log_interval: int = 10
    sample_interval: int = 1000      # generate a sample every N steps (0 disables)
    seed: int = 1337
    device: str = "auto"             # "auto" | "cuda" | "mps" | "cpu"
    dtype: str = "auto"              # "auto" | "bfloat16" | "float16" | "float32"
    compile: bool = False            # torch.compile (off by default for clarity)
    # effective tokens/optimizer-step = batch_size * grad_accum * context_length


@dataclass
class CheckpointConfig:
    out_dir: str = "checkpoints/base"
    save_interval: int = 1000        # save a "last" checkpoint every N steps
    keep_best: bool = True           # also track best-val checkpoint separately
    resume: bool = False             # resume from out_dir/last.pt if present


@dataclass
class LoggingConfig:
    wandb_enabled: bool = False
    wandb_project: str = "hindi-llm"
    wandb_run_name: str | None = None
    # A local JSONL metrics log is always written, regardless of wandb.
    metrics_jsonl: str = "outputs/metrics.jsonl"


@dataclass
class SFTConfig:
    """Supervised fine-tuning of the pretrained base model on chat data."""

    data_path: str = "data/sample_sft.jsonl"
    base_checkpoint: str = "checkpoints/base/best.pt"
    out_dir: str = "checkpoints/sft"
    epochs: int = 3
    lr: float = 1e-5                 # much smaller than pretraining LR
    batch_size: int = 8
    grad_accum_steps: int = 1
    warmup_steps: int = 20
    max_seq_len: int = 1024
    mask_prompt: bool = True         # train loss only on assistant tokens
    weight_decay: float = 0.0


@dataclass
class EvalConfig:
    base_checkpoint: str = "checkpoints/base/best.pt"
    sft_checkpoint: str = "checkpoints/sft/best.pt"
    out_markdown: str = "outputs/eval_report.md"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    # Hindi prompts used for qualitative generation.
    prompts: list[str] = field(
        default_factory=lambda: [
            "भारत एक ऐसा देश है जहाँ",
            "विज्ञान का महत्व यह है कि",
            "सुबह जल्दी उठने के फायदे हैं",
            "एक बार की बात है, एक छोटे से गाँव में",
        ]
    )


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """The single root config object, composed of the groups above."""

    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a YAML file of overrides merged on top of the defaults."""
        with Path(path).open("r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        return cls.from_dict(overrides)

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "Config":
        cfg = cls()
        _merge_into_dataclass(cfg, overrides)
        # keep the duplicated vocab_size in sync (tokenizer drives the model)
        cfg.model.vocab_size = cfg.tokenizer.vocab_size
        return cfg

    def sync_vocab(self, vocab_size: int) -> None:
        """Force both tokenizer and model to agree on the real vocab size.

        The trained tokenizer is the ground truth (BPE may stop short of the
        requested size on tiny corpora), so callers pass the actual size here.
        """
        self.tokenizer.vocab_size = vocab_size
        self.model.vocab_size = vocab_size


def _merge_into_dataclass(obj: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply a (possibly partial) dict onto a dataclass instance."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(
                f"Unknown config key '{key}' for {type(obj).__name__}. "
                f"Valid keys: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(obj, key, value)


# --------------------------------------------------------------------------- #
# Parameter-count estimator
# --------------------------------------------------------------------------- #
def estimate_num_params(m: ModelConfig) -> dict[str, int]:
    """Return a breakdown of parameter counts for a given model config.

    The arithmetic mirrors the modules in model.py exactly (no biases, RMSNorm
    has one weight per feature, SwiGLU has 3 matrices). Useful both for the
    README's "~50M" claim and as a sanity check against the built model.
    """
    V, d, L = m.vocab_size, m.d_model, m.n_layers
    h = m.resolve_mlp_hidden()

    embedding = V * d                       # token embedding table
    # attention: q, k, v, o projections (square, no bias)
    attn_per_layer = 4 * d * d
    if m.attn_bias:
        attn_per_layer += 4 * d
    # SwiGLU MLP: gate (d->h), up (d->h), down (h->d)
    mlp_per_layer = 3 * d * h
    if m.mlp_bias:
        mlp_per_layer += 2 * h + d
    norms_per_layer = 2 * d                 # two RMSNorms (attn + mlp), weight only
    per_layer = attn_per_layer + mlp_per_layer + norms_per_layer

    blocks = L * per_layer
    final_norm = d
    # output projection: tied -> reuses embedding (0 extra); untied -> V*d
    head = 0 if m.tie_embeddings else V * d

    total = embedding + blocks + final_norm + head
    return {
        "embedding": embedding,
        "per_layer": per_layer,
        "blocks_total": blocks,
        "final_norm": final_norm,
        "lm_head": head,
        "total": total,
        "non_embedding": total - embedding - head,
    }


def print_param_count(cfg: Config | ModelConfig | None = None) -> int:
    """Pretty-print the parameter breakdown; returns the total count."""
    m = cfg.model if isinstance(cfg, Config) else (cfg or ModelConfig())
    b = estimate_num_params(m)

    def fmt(n: int) -> str:
        return f"{n:>13,} ({n / 1e6:6.2f}M)"

    print("Model configuration:")
    print(f"  vocab_size      : {m.vocab_size}")
    print(f"  context_length  : {m.context_length}")
    print(f"  d_model         : {m.d_model}")
    print(f"  n_layers        : {m.n_layers}")
    print(f"  n_heads         : {m.n_heads} (head_dim={m.head_dim})")
    print(f"  mlp_hidden      : {m.resolve_mlp_hidden()} (SwiGLU)")
    print(f"  tie_embeddings  : {m.tie_embeddings}")
    print("Parameter breakdown:")
    print(f"  embedding       : {fmt(b['embedding'])}")
    print(f"  per transformer : {fmt(b['per_layer'])}  x {m.n_layers} layers")
    print(f"  blocks total    : {fmt(b['blocks_total'])}")
    print(f"  final norm      : {fmt(b['final_norm'])}")
    print(f"  lm_head         : {fmt(b['lm_head'])}")
    print(f"  non-embedding   : {fmt(b['non_embedding'])}")
    print(f"  TOTAL           : {fmt(b['total'])}")
    return b["total"]


if __name__ == "__main__":
    # `python -m hindi_llm.config` prints the default 50M breakdown.
    print_param_count(Config())
