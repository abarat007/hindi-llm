#!/usr/bin/env python3
"""Phase 4 — pretraining loop for the Hindi GPT.

A hand-written training loop (no HF Trainer): bf16 autocast when the GPU
supports it (fp32 fallback), gradient accumulation, AdamW, cosine LR schedule
with warmup, gradient clipping, periodic validation perplexity, checkpointing
with resume and best-tracking, periodic Hindi sample generations, a local JSONL
metrics log (always on) and optional Weights & Biases.

Config comes from a YAML (default ``configs/hindi_50m.yaml``); any field can be
overridden on the command line with repeated ``--set dotted.key=value`` flags,
e.g. for a tiny CPU smoke run:

    python scripts/train.py --config configs/hindi_50m.yaml \
        --set model.context_length=64 --set model.d_model=128 \
        --set model.n_layers=2 --set model.n_heads=4 \
        --set train.batch_size=8 --set train.grad_accum_steps=1 \
        --set train.max_steps=20 --set train.eval_interval=10 \
        --set train.sample_interval=0

Resume an interrupted run:

    python scripts/train.py --config configs/hindi_50m.yaml --set checkpoint.resume=true
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.config import Config  # noqa: E402
from hindi_llm.data import BinDataset, load_meta  # noqa: E402
from hindi_llm.model import build_model  # noqa: E402
from hindi_llm.sampling import generate  # noqa: E402
from hindi_llm.tokenizer_io import HindiTokenizer  # noqa: E402
from hindi_llm import train_utils as tu  # noqa: E402


# --------------------------------------------------------------------------- #
# CLI / config override helpers
# --------------------------------------------------------------------------- #
def set_by_path(cfg: Config, dotted: str, value) -> None:
    """Assign cfg.a.b.c = value given the dotted path 'a.b.c'."""
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown config path: {dotted}")
    setattr(obj, parts[-1], value)


def apply_overrides(cfg: Config, overrides: list[str]) -> None:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        # yaml parses ints/floats/bools/null with correct types
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pretrain the Hindi GPT from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/hindi_50m.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Override any config field (repeatable).")
    p.add_argument("--resume", action="store_true", help="Shortcut for checkpoint.resume=true.")
    p.add_argument("--wandb", action="store_true", help="Shortcut for logging.wandb_enabled=true.")
    return p


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def load_config(args) -> Config:
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    apply_overrides(cfg, args.overrides)
    if args.resume:
        cfg.checkpoint.resume = True
    if args.wandb:
        cfg.logging.wandb_enabled = True
    return cfg


def maybe_init_wandb(cfg: Config):
    if not cfg.logging.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb requested but not installed; continuing without it.",
              file=sys.stderr)
        return None
    wandb.init(project=cfg.logging.wandb_project, name=cfg.logging.wandb_run_name,
               config=cfg.to_dict())
    return wandb


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args)
    tu.set_seed(cfg.train.seed)

    # --- data ---------------------------------------------------------------
    meta_path = Path(cfg.data.processed_dir) / "meta.json"
    if not meta_path.exists():
        print(f"[error] {meta_path} not found. Run encode_dataset.py first.",
              file=sys.stderr)
        return 1
    meta = load_meta(meta_path)
    cfg.sync_vocab(meta["vocab_size"])  # tokenizer is the source of truth
    import numpy as np
    dt = np.dtype(meta["dtype"])
    train_ds = BinDataset(cfg.data.train_bin, dtype=dt)
    val_ds = BinDataset(cfg.data.val_bin, dtype=dt)
    datasets = {"train": train_ds, "val": val_ds}

    # --- device / precision -------------------------------------------------
    device = tu.resolve_device(cfg.train.device)
    amp_dtype = tu.resolve_amp_dtype(cfg.train.dtype, device)
    print(f"device={device}  amp_dtype={amp_dtype}  "
          f"train_tokens={len(train_ds):,}  val_tokens={len(val_ds):,}")

    block_size = cfg.model.context_length
    if len(train_ds) <= block_size + 1:
        new_bs = max(8, len(train_ds) // 4)
        print(f"[warn] tiny corpus; reducing block_size {block_size} -> {new_bs}")
        block_size = new_bs
        cfg.model.context_length = block_size

    # --- model / optimizer --------------------------------------------------
    model = build_model(cfg.model).to(device)
    print(f"model parameters: {model.num_params():,} "
          f"({model.num_params() / 1e6:.2f}M); "
          f"non-embedding: {model.num_params(non_embedding=True) / 1e6:.2f}M")
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay, lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
    )
    scaler = (torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
              if device == "cuda" else None)

    # --- resume -------------------------------------------------------------
    out_dir = Path(cfg.checkpoint.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_val = float("inf")
    last_ckpt = out_dir / "last.pt"
    if cfg.checkpoint.resume and last_ckpt.exists():
        ck = tu.load_checkpoint(last_ckpt, model, optimizer, scaler, device)
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {last_ckpt} at step {start_step} (best_val={best_val:.4f})")

    # snapshot the exact config used for this run
    cfg.to_yaml(out_dir / "config_snapshot.yaml")
    logger = tu.JsonlLogger(cfg.logging.metrics_jsonl)
    wandb = maybe_init_wandb(cfg)

    # --- optional tokenizer for periodic samples ----------------------------
    tok = None
    if cfg.train.sample_interval and Path(cfg.tokenizer.path).exists():
        tok = HindiTokenizer.load(cfg.tokenizer.path)

    lr_decay_steps = cfg.scheduler.lr_decay_steps or cfg.train.max_steps
    ga = cfg.train.grad_accum_steps
    tokens_per_step = cfg.train.batch_size * ga * block_size

    model.train()
    print(f"starting training: {cfg.train.max_steps} steps, "
          f"{tokens_per_step:,} tokens/step")
    t0 = time.time()

    for step in range(start_step, cfg.train.max_steps):
        # set LR for this step
        lr = tu.get_lr(step, cfg.scheduler.warmup_steps, lr_decay_steps,
                       cfg.optim.lr, cfg.scheduler.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- gradient accumulation over micro-steps ----
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(ga):
            x, y = train_ds.get_batch(cfg.train.batch_size, block_size, device)
            with tu.autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)
                loss = loss / ga                # scale so grads average correctly
            scaler.scale(loss).backward() if scaler else loss.backward()
            accum_loss += loss.item()

        # ---- clip + step ----
        if scaler:
            scaler.unscale_(optimizer)
        if cfg.optim.grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optim.grad_clip).item()
        else:
            gnorm = tu.grad_global_norm(model)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ---- logging ----
        if step % cfg.train.log_interval == 0:
            dt_s = time.time() - t0
            tps = tokens_per_step * cfg.train.log_interval / dt_s if step else tokens_per_step / dt_s
            rec = {"step": step, "train_loss": accum_loss, "lr": lr,
                   "grad_norm": gnorm, "tokens_per_sec": tps}
            logger.log(rec)
            if wandb:
                wandb.log(rec, step=step)
            print(f"step {step:>6} | loss {accum_loss:7.4f} | lr {lr:.2e} "
                  f"| gnorm {gnorm:6.2f} | {tps/1e3:7.1f}K tok/s")
            t0 = time.time()

        # ---- eval + checkpoint ----
        if step > 0 and step % cfg.train.eval_interval == 0:
            stats = tu.estimate_loss(model, datasets, cfg.train.eval_iters,
                                     cfg.train.batch_size, block_size, device, amp_dtype)
            vloss = stats["val"]["loss"]
            rec = {"step": step, "eval_train_loss": stats["train"]["loss"],
                   "eval_val_loss": vloss, "val_ppl": stats["val"]["ppl"]}
            logger.log(rec)
            if wandb:
                wandb.log(rec, step=step)
            print(f"  eval @ {step}: train {stats['train']['loss']:.4f} | "
                  f"val {vloss:.4f} | val ppl {stats['val']['ppl']:.2f}")
            tu.save_checkpoint(out_dir / "last.pt", model, optimizer, step,
                               best_val, cfg.to_dict(), scaler)
            if cfg.checkpoint.keep_best and vloss < best_val:
                best_val = vloss
                tu.save_checkpoint(out_dir / "best.pt", model, optimizer, step,
                                   best_val, cfg.to_dict(), scaler)
                print(f"  new best val loss {best_val:.4f} -> best.pt")

        # ---- periodic sample ----
        if tok and cfg.train.sample_interval and step > 0 \
                and step % cfg.train.sample_interval == 0:
            _sample(model, tok, device)

        # ---- periodic "last" save ----
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            tu.save_checkpoint(out_dir / "last.pt", model, optimizer, step,
                               best_val, cfg.to_dict(), scaler)

    # final save
    tu.save_checkpoint(out_dir / "last.pt", model, optimizer,
                       cfg.train.max_steps, best_val, cfg.to_dict(), scaler)
    if not (out_dir / "best.pt").exists():
        # ensure a best.pt exists even for very short runs
        tu.save_checkpoint(out_dir / "best.pt", model, optimizer,
                           cfg.train.max_steps, best_val, cfg.to_dict(), scaler)
    logger.close()
    print(f"\ntraining complete. checkpoints in {out_dir}")
    return 0


def _sample(model, tok: HindiTokenizer, device: str) -> None:
    prompt = "भारत एक ऐसा देश है"
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
    out = generate(model, ids, max_new_tokens=40, temperature=0.8, top_k=40,
                   eos_id=tok.eos_id)
    text = tok.decode(out[0].tolist())
    print(f"  sample: {text}")


if __name__ == "__main__":
    raise SystemExit(main())
