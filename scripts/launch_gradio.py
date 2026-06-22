#!/usr/bin/env python3
"""Phase 6 — minimal Gradio chat UI for the Hindi SFT model.

Loads an SFT checkpoint, applies the chat template, and streams a response.
Temperature / top-k / max-new-tokens are exposed as sliders.

This is a tiny research demo, NOT a production assistant — the banner says so.

Example:
    python scripts/launch_gradio.py --checkpoint checkpoints/sft/best.pt
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

WARNING = (
    "⚠️ यह एक छोटा शोध-प्रोटोटाइप है (≈50M पैरामीटर), उत्पादन-स्तर का सहायक नहीं। "
    "उत्तर अक्सर गलत या अर्थहीन हो सकते हैं।\n\n"
    "**Warning:** tiny (~50M) research demo trained from scratch — answers are "
    "frequently wrong or nonsensical. Do not rely on them."
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch the Hindi chat demo.")
    p.add_argument("--checkpoint", default="checkpoints/sft/best.pt")
    p.add_argument("--tokenizer", default="tokenizer/hindi_bpe.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        import gradio as gr
    except ImportError:
        print("[error] gradio is not installed. Install the demo extra:\n"
              "  uv pip install -e '.[demo]'   (or: pip install gradio)",
              file=sys.stderr)
        return 1

    device = tu.resolve_device(args.device)
    tok = HindiTokenizer.load(args.tokenizer)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {device} "
          f"({model.num_params() / 1e6:.2f}M params)")

    @torch.no_grad()
    def respond(message: str, history, temperature: float, top_k: int,
                max_new_tokens: int) -> str:
        if not message.strip():
            return ""
        ids, _ = build_chat_ids(tok, [Turn("user", message)],
                                add_generation_prompt=True)
        prompt = torch.tensor([ids], device=device)
        out = generate(model, prompt, int(max_new_tokens),
                       temperature=float(temperature), top_k=int(top_k),
                       eos_id=tok.eos_id)
        return tok.decode(out[0, len(ids):].tolist()).strip()

    with gr.Blocks(title="Hindi LLM (from scratch)") as demo:
        gr.Markdown("# हिंदी भाषा मॉडल — डेमो")
        gr.Markdown(WARNING)
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_k = gr.Slider(0, 200, value=40, step=1, label="top-k")
            max_new = gr.Slider(16, 256, value=128, step=8, label="max new tokens")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[temperature, top_k, max_new],
            examples=[["भारत की राजधानी क्या है?"], ["सूरज किस दिशा से उगता है?"]],
        )

    demo.launch(share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
