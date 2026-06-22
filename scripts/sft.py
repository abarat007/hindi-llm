#!/usr/bin/env python3
"""Phase 5 — supervised fine-tuning (SFT) of the Hindi base model into a chat model.

Loads a pretrained *base* checkpoint, fine-tunes it on a small instruction/chat
dataset using the project's chat template, and writes a *separate* SFT
checkpoint (base and chat models are kept apart on purpose — see
docs/training_notes.md). Loss is masked so gradients flow only through the
assistant's tokens (and its terminating <eos>), not the prompt.

Supported JSONL example formats (one object per line):
  * {"system": ..., "user": ..., "assistant": ...}
  * {"instruction": ..., "input": ..., "output": ...}   (input optional)
  * {"messages": [{"role": "user", "content": ...}, ...]}

Right-padding is safe without an attention mask: the model is causal and pads
sit at the end, so real tokens (which precede them) never attend to pads, and
the pad targets are -100 (ignored).

Example:
    python scripts/sft.py --config configs/hindi_50m.yaml \
        --set sft.data_path=data/sample_sft.jsonl \
        --set sft.base_checkpoint=checkpoints/base/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.config import Config, ModelConfig  # noqa: E402
from hindi_llm.chat_template import Turn, build_chat_ids  # noqa: E402
from hindi_llm.model import build_model  # noqa: E402
from hindi_llm.sampling import generate  # noqa: E402
from hindi_llm.tokenizer_io import HindiTokenizer  # noqa: E402
from hindi_llm import train_utils as tu  # noqa: E402

# reuse the train script's override helpers
import train  # noqa: E402


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def record_to_messages(rec: dict) -> list[Turn]:
    """Map one JSONL record (any supported schema) to a list of Turns."""
    if "messages" in rec:
        return [Turn(m["role"], m["content"]) for m in rec["messages"]]
    if "user" in rec and "assistant" in rec:
        turns = []
        if rec.get("system"):
            turns.append(Turn("system", rec["system"]))
        turns.append(Turn("user", rec["user"]))
        turns.append(Turn("assistant", rec["assistant"]))
        return turns
    if "instruction" in rec and "output" in rec:
        user = rec["instruction"]
        if rec.get("input"):
            user = f"{user}\n{rec['input']}"
        return [Turn("user", user), Turn("assistant", rec["output"])]
    raise ValueError(f"unrecognized SFT record schema: {sorted(rec)}")


def load_examples(
    path: Path, tok: HindiTokenizer, max_seq_len: int, mask_prompt: bool
) -> list[tuple[list[int], list[int]]]:
    """Return a list of (x_ids, y_labels) where y is masked with -100 outside
    the trained region."""
    if not path.exists():
        raise FileNotFoundError(f"SFT data not found: {path}")
    examples: list[tuple[list[int], list[int]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = record_to_messages(rec)
            ids, amask = build_chat_ids(tok, messages, add_generation_prompt=False)
            if not mask_prompt:
                amask = [True] * len(ids)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                amask = amask[:max_seq_len]
            # next-token targets: predict ids[t+1] from position t
            x = ids[:-1]
            y = [ids[t + 1] if amask[t + 1] else -100 for t in range(len(ids) - 1)]
            if any(t != -100 for t in y):  # skip examples with nothing to learn
                examples.append((x, y))
    return examples


def make_batch(
    batch: list[tuple[list[int], list[int]]], pad_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch to its longest sequence."""
    maxlen = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = maxlen - len(x)
        xs.append(x + [pad_id] * pad)        # pad inputs with <pad>
        ys.append(y + [-100] * pad)          # ignore padded targets
    x_t = torch.tensor(xs, dtype=torch.long, device=device)   # [B, T]
    y_t = torch.tensor(ys, dtype=torch.long, device=device)   # [B, T]
    return x_t, y_t


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supervised fine-tune the Hindi base model into a chat model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/hindi_50m.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    train.apply_overrides(cfg, args.overrides)
    tu.set_seed(cfg.train.seed)

    device = tu.resolve_device(cfg.train.device)
    amp_dtype = tu.resolve_amp_dtype(cfg.train.dtype, device)

    tok = HindiTokenizer.load(cfg.tokenizer.path)

    # --- rebuild the base model from its own saved config, then load weights --
    base_path = Path(cfg.sft.base_checkpoint)
    if not base_path.exists():
        print(f"[error] base checkpoint not found: {base_path}. Pretrain first "
              f"with scripts/train.py.", file=sys.stderr)
        return 1
    # load the raw checkpoint first so we can rebuild the exact base architecture
    # from its saved config before loading the weights into it
    ckpt = torch.load(base_path, map_location=device, weights_only=False)
    base_model_cfg = ModelConfig(**ckpt["config"]["model"])
    base_model_cfg.vocab_size = tok.vocab_size
    model = build_model(base_model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded base model from {base_path} "
          f"({model.num_params() / 1e6:.2f}M params)")

    # --- data ---------------------------------------------------------------
    # never exceed the context the base model was actually trained with
    max_seq_len = min(cfg.sft.max_seq_len, base_model_cfg.context_length)
    examples = load_examples(Path(cfg.sft.data_path), tok, max_seq_len,
                             cfg.sft.mask_prompt)
    if not examples:
        print("[error] no usable SFT examples found.", file=sys.stderr)
        return 1
    print(f"SFT examples: {len(examples)} | mask_prompt={cfg.sft.mask_prompt}")

    bs = cfg.sft.batch_size
    steps_per_epoch = math.ceil(len(examples) / bs)
    total_steps = steps_per_epoch * cfg.sft.epochs

    optimizer = model.configure_optimizers(
        weight_decay=cfg.sft.weight_decay, lr=cfg.sft.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
    )
    scaler = (torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
              if device == "cuda" else None)

    out_dir = Path(cfg.sft.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config_snapshot.yaml")
    logger = tu.JsonlLogger(str(out_dir / "sft_metrics.jsonl"))

    rng = torch.Generator().manual_seed(cfg.train.seed)
    model.train()
    step = 0
    best = float("inf")
    print(f"starting SFT: {cfg.sft.epochs} epochs x {steps_per_epoch} steps "
          f"= {total_steps} steps")

    for epoch in range(cfg.sft.epochs):
        # shuffle example order each epoch
        order = torch.randperm(len(examples), generator=rng).tolist()
        epoch_loss = 0.0
        for bi in range(steps_per_epoch):
            batch_idx = order[bi * bs:(bi + 1) * bs]
            batch = [examples[i] for i in batch_idx]
            x, y = make_batch(batch, tok.pad_id, device)

            lr = tu.get_lr(step, cfg.sft.warmup_steps, total_steps, cfg.sft.lr,
                           cfg.sft.lr * 0.1)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with tu.autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)
            (scaler.scale(loss).backward() if scaler else loss.backward())
            if scaler:
                scaler.unscale_(optimizer)
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()

            epoch_loss += loss.item()
            logger.log({"step": step, "epoch": epoch, "loss": loss.item(), "lr": lr})
            step += 1

        avg = epoch_loss / steps_per_epoch
        print(f"epoch {epoch}: avg loss {avg:.4f} | ppl {math.exp(min(avg,20)):.2f}")
        tu.save_checkpoint(out_dir / "last.pt", model, optimizer, step, best,
                           cfg.to_dict(), scaler)
        if avg < best:
            best = avg
            tu.save_checkpoint(out_dir / "best.pt", model, optimizer, step, best,
                               cfg.to_dict(), scaler)

    logger.close()
    print(f"\nSFT complete. checkpoints in {out_dir}")

    # --- sanity sample ------------------------------------------------------
    _sanity_sample(model, tok, device)
    return 0


def _sanity_sample(model, tok: HindiTokenizer, device: str) -> None:
    print("\n=== SFT sanity sample ===")
    msgs = [Turn("user", "भारत की राजधानी क्या है?")]
    ids, _ = build_chat_ids(tok, msgs, add_generation_prompt=True)
    prompt = torch.tensor([ids], device=device)
    out = generate(model, prompt, max_new_tokens=64, temperature=0.7, top_k=40,
                   eos_id=tok.eos_id)
    # decode only the newly generated assistant span
    gen = out[0, len(ids):].tolist()
    print("user: भारत की राजधानी क्या है?")
    print("assistant:", tok.decode(gen).strip())


if __name__ == "__main__":
    raise SystemExit(main())
