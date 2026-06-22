#!/usr/bin/env python3
"""Phase 6 — evaluate the Hindi model(s) and write a qualitative report.

Computes deterministic validation perplexity and qualitative generations for the
base and/or SFT checkpoints (whichever exist), and writes a Markdown report with
the Hindi prompts, the generations, and explicit failure-mode notes.

We never invent metrics: numbers come from the actual checkpoint + val shard you
point at. If a checkpoint or val shard is missing, that section is skipped with a
clear note.

Example:
    python scripts/evaluate.py --config configs/hindi_50m.yaml \
        --base-checkpoint checkpoints/base/best.pt \
        --sft-checkpoint checkpoints/sft/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.config import Config  # noqa: E402
from hindi_llm.data import BinDataset, load_meta  # noqa: E402
from hindi_llm.eval_utils import (  # noqa: E402
    chat_responses,
    complete_prompts,
    corpus_perplexity,
    load_model_from_checkpoint,
)
from hindi_llm.tokenizer_io import HindiTokenizer  # noqa: E402
from hindi_llm import train_utils as tu  # noqa: E402


# A small fixed set of Hindi chat prompts for qualitative SFT evaluation.
CHAT_PROMPTS = [
    "भारत की राजधानी क्या है?",
    "पानी की तीन अवस्थाएँ कौन-सी हैं?",
    "स्वस्थ रहने के लिए क्या करना चाहिए?",
    "सूरज किस दिशा से उगता है?",
]

FAILURE_MODE_NOTES = """\
## Known failure modes (read these honestly)

A ~50M model trained on a modest Hindi corpus is a *research toy*, not a usable
assistant. Expect:

- **Repetition / loops** — the model may repeat words or phrases, especially at
  low temperature or with a weak base. Mitigate with top-p / higher temperature.
- **Code-switching** — stray English/Latin fragments if the corpus was not
  cleaned aggressively (see the Latin-ratio filter in prepare_corpus.py).
- **Factual errors / hallucination** — there is far too little training signal
  for reliable facts; treat every claim as unreliable.
- **Run-on or truncated output** — weak end-of-sequence behavior if SFT data was
  tiny; the model may not learn to stop.
- **Devanagari spelling slips** — matra/akshara errors when the tokenizer split
  a rare word into many pieces (high local fertility).

These are expected at this scale and improve with more/cleaner data, more
parameters, and more SFT examples (see docs/scaling_10x.md).
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate base/SFT Hindi checkpoints; write a Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/hindi_50m.yaml")
    p.add_argument("--tokenizer", default=None, help="Override cfg.tokenizer.path.")
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--sft-checkpoint", default=None)
    p.add_argument("--out", default=None, help="Output markdown path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-windows", type=int, default=200,
                   help="Cap val windows for perplexity (None-like for full).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    device = tu.resolve_device(args.device)

    tok_path = args.tokenizer or cfg.tokenizer.path
    tok = HindiTokenizer.load(tok_path)

    base_ckpt = args.base_checkpoint or cfg.eval.base_checkpoint
    sft_ckpt = args.sft_checkpoint or cfg.eval.sft_checkpoint
    out_path = Path(args.out or cfg.eval.out_markdown)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # validation dataset (optional — only needed for perplexity)
    val_ds = None
    meta_path = Path(cfg.data.processed_dir) / "meta.json"
    if Path(cfg.data.val_bin).exists() and meta_path.exists():
        meta = load_meta(meta_path)
        val_ds = BinDataset(cfg.data.val_bin, dtype=np.dtype(meta["dtype"]))

    lines: list[str] = ["# Hindi LLM — evaluation report", ""]
    lines.append(f"- device: `{device}`")
    lines.append(f"- tokenizer: `{tok_path}` (vocab {tok.vocab_size})")
    lines.append("")

    # ---- base model --------------------------------------------------------
    if Path(base_ckpt).exists():
        print(f"evaluating base checkpoint: {base_ckpt}")
        model, _ = load_model_from_checkpoint(base_ckpt, device)
        lines.append("## Base model")
        lines.append("")
        lines.append(f"- checkpoint: `{base_ckpt}`")
        lines.append(f"- parameters: {model.num_params() / 1e6:.2f}M")
        if val_ds is not None:
            mw = None if args.max_windows < 0 else args.max_windows
            ppl = corpus_perplexity(model, val_ds, model.cfg.context_length,
                                    device, max_windows=mw)
            lines.append(f"- validation perplexity: **{ppl['ppl']:.2f}** "
                         f"(loss {ppl['loss']:.4f} over {ppl['tokens']:,} tokens)")
            print(f"  base val ppl: {ppl['ppl']:.2f}")
        else:
            lines.append("- validation perplexity: _skipped (no val shard found)_")
        lines.append("")
        lines.append("### Base completions (free-form)")
        lines.append("")
        comps = complete_prompts(model, tok, cfg.eval.prompts, device,
                                 max_new_tokens=cfg.eval.max_new_tokens,
                                 temperature=cfg.eval.temperature,
                                 top_k=cfg.eval.top_k, top_p=cfg.eval.top_p)
        for c in comps:
            lines.append(f"- **prompt:** {c['prompt']}")
            lines.append(f"  - **output:** {c['completion']}")
        lines.append("")
    else:
        lines.append("## Base model\n\n_skipped: checkpoint "
                     f"`{base_ckpt}` not found._\n")

    # ---- SFT model ---------------------------------------------------------
    if Path(sft_ckpt).exists():
        print(f"evaluating SFT checkpoint: {sft_ckpt}")
        model, _ = load_model_from_checkpoint(sft_ckpt, device)
        lines.append("## SFT chat model")
        lines.append("")
        lines.append(f"- checkpoint: `{sft_ckpt}`")
        lines.append("")
        lines.append("### Chat responses")
        lines.append("")
        chats = chat_responses(model, tok, CHAT_PROMPTS, device,
                               max_new_tokens=cfg.eval.max_new_tokens,
                               temperature=cfg.eval.temperature,
                               top_k=cfg.eval.top_k, top_p=cfg.eval.top_p)
        for c in chats:
            lines.append(f"- **user:** {c['user']}")
            lines.append(f"  - **assistant:** {c['assistant']}")
        lines.append("")
    else:
        lines.append("## SFT chat model\n\n_skipped: checkpoint "
                     f"`{sft_ckpt}` not found._\n")

    lines.append(FAILURE_MODE_NOTES)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote evaluation report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
