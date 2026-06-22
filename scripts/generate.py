#!/usr/bin/env python3
"""Generate Hindi text from a trained checkpoint.

Works with either a base checkpoint (free-form continuation of a prompt) or an
SFT checkpoint (chat-formatted response). Exposes temperature, top-k, top-p,
max-new-tokens and a seed.

Examples:
    # base model: continue a prompt
    python scripts/generate.py --checkpoint checkpoints/base/best.pt \
        --prompt "भारत एक ऐसा देश है" --max-new-tokens 80

    # SFT model: chat
    python scripts/generate.py --checkpoint checkpoints/sft/best.pt --chat \
        --prompt "भारत की राजधानी क्या है?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.chat_template import Turn, build_chat_ids  # noqa: E402
from hindi_llm.eval_utils import load_model_from_checkpoint  # noqa: E402
from hindi_llm.sampling import generate  # noqa: E402
from hindi_llm.tokenizer_io import HindiTokenizer  # noqa: E402
from hindi_llm import train_utils as tu  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Hindi text from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="tokenizer/hindi_bpe.json")
    p.add_argument("--prompt", required=True, help="Hindi prompt / user message.")
    p.add_argument("--chat", action="store_true",
                   help="Apply the chat template (use for SFT checkpoints).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = tu.resolve_device(args.device)
    tok = HindiTokenizer.load(args.tokenizer)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)

    gen = tu.make_generator(device, args.seed)
    if args.chat:
        ids, _ = build_chat_ids(tok, [Turn("user", args.prompt)],
                                add_generation_prompt=True)
        prompt_ids = torch.tensor([ids], device=device)
        prompt_len = len(ids)
    else:
        ids = tok.encode(args.prompt, add_bos=True)
        prompt_ids = torch.tensor([ids], device=device)
        prompt_len = 0  # show the whole continuation including the prompt

    out = generate(
        model, prompt_ids, args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, eos_id=tok.eos_id, generator=gen,
    )
    text = tok.decode(out[0, prompt_len:].tolist()).strip()

    print("=" * 60)
    if args.chat:
        print(f"user: {args.prompt}")
        print(f"assistant: {text}")
    else:
        print(text)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
