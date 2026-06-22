#!/usr/bin/env python3
"""Phase 2 — train a Hindi BPE tokenizer with the `tokenizers` library.

We call the low-level `tokenizers` API directly (no HuggingFace Trainer). The
pipeline is:

    NFC normalize  ->  Metaspace pre-tokenize  ->  BPE  ->  Metaspace decode

Why these choices (see docs/tokenizer_notes.md for detail):

  * NFC normalization gives Devanagari a single canonical code-point form, so
    visually identical aksharas don't fork into separate vocabulary entries.
  * Metaspace (the SentencePiece "▁" word-boundary marker) handles spacing
    losslessly and reversibly, which suits Hindi's space-separated words.
  * BPE with an `<unk>` token; because the corpus alphabet (Devanagari + a little
    Latin/punct) is small, virtually every character is learned, so the measured
    unknown-token rate is reported (and should be ~0) rather than assumed.

It also writes a tokenizer report (fertility, compression, unknown rate, vocab
size, examples) and compares against a baseline — an external tokenizer JSON if
you pass one, otherwise clearly-labeled byte-level and char-level fallbacks.

Example:
    python scripts/train_tokenizer.py \
        --input data/processed/clean.txt \
        --output tokenizer/hindi_bpe.json --vocab-size 32000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# Import the canonical special-token list from the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.config import SPECIAL_TOKENS  # noqa: E402


# --------------------------------------------------------------------------- #
# Corpus iteration
# --------------------------------------------------------------------------- #
def gather_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.suffix in (".txt", ".jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] input not found: {p}", file=sys.stderr)
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl inputs found in: {inputs}")
    return files


def iter_lines(files: list[Path], text_field: str) -> Iterator[str]:
    """Yield text lines from .txt and .jsonl files for training."""
    for path in files:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and text_field in obj:
                        yield str(obj[text_field])
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n")


# --------------------------------------------------------------------------- #
# Build + train tokenizer
# --------------------------------------------------------------------------- #
def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFC()])
    # Metaspace marks word starts with ▁ and is fully reversible.
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    return tok


def train_tokenizer(
    files: list[Path],
    vocab_size: int,
    min_frequency: int,
    text_field: str,
) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        # Hindi's base alphabet is small; a generous cap ensures every
        # Devanagari character becomes part of the vocabulary (low UNK).
        limit_alphabet=1000,
        show_progress=True,
    )
    tok.train_from_iterator(iter_lines(files, text_field), trainer=trainer)
    return tok


# --------------------------------------------------------------------------- #
# Metrics + report
# --------------------------------------------------------------------------- #
def compute_metrics(tok: Tokenizer, sample: list[str]) -> dict:
    """Fertility, compression, and unknown-token rate over a text sample."""
    unk_id = tok.token_to_id("<unk>")
    n_tokens = n_words = n_chars = n_unk = 0
    for line in sample:
        n_chars += len(line)
        n_words += len(line.split())
        ids = tok.encode(line).ids
        n_tokens += len(ids)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
    n_tokens = max(n_tokens, 1)
    return {
        "sample_lines": len(sample),
        "sample_chars": n_chars,
        "sample_words": n_words,
        "sample_tokens": n_tokens,
        # fertility: tokens emitted per whitespace word (lower is better)
        "fertility_tokens_per_word": n_tokens / max(n_words, 1),
        # compression: characters represented per token (higher is better)
        "compression_chars_per_token": n_chars / n_tokens,
        "unknown_token_rate": n_unk / n_tokens,
    }


def baseline_metrics(sample: list[str], baseline_tok_path: str | None) -> dict:
    """Comparison baseline.

    If an external tokenizer JSON is supplied and loads, use it. Otherwise fall
    back to byte-level and char-level *estimates*, clearly labeled as fallbacks
    (we never pretend a real external baseline exists).
    """
    if baseline_tok_path:
        p = Path(baseline_tok_path)
        if p.exists():
            try:
                bt = Tokenizer.from_file(str(p))
                m = compute_metrics(bt, sample)
                m["kind"] = f"external:{p.name}"
                return m
            except Exception as e:  # noqa: BLE001
                print(f"[warn] could not load baseline tokenizer: {e}",
                      file=sys.stderr)
        else:
            print(f"[warn] baseline tokenizer not found: {p}", file=sys.stderr)

    # Fallback estimates (NOT a real trained baseline).
    n_words = sum(len(l.split()) for l in sample)
    n_chars = sum(len(l) for l in sample)
    n_bytes = sum(len(l.encode("utf-8")) for l in sample)
    return {
        "kind": "fallback (byte-level & char-level estimate, NOT a trained tokenizer)",
        "char_level": {
            "fertility_tokens_per_word": n_chars / max(n_words, 1),
            "compression_chars_per_token": 1.0,
        },
        "byte_level_utf8": {
            "fertility_tokens_per_word": n_bytes / max(n_words, 1),
            "compression_chars_per_token": n_chars / max(n_bytes, 1),
        },
    }


def tokenization_examples(tok: Tokenizer, sentences: list[str]) -> list[dict]:
    out = []
    for s in sentences:
        enc = tok.encode(s)
        out.append({
            "text": s,
            "n_tokens": len(enc.ids),
            "tokens": enc.tokens,
        })
    return out


_DEFAULT_EXAMPLES = [
    "भारत एक विशाल और विविधताओं से भरा देश है।",
    "मशीन लर्निंग आजकल बहुत तेज़ी से आगे बढ़ रहा है।",
    "नमस्ते, आप कैसे हैं?",
    "गंगा नदी हिमालय से निकलकर बंगाल की खाड़ी में मिलती है।",
]


def write_report(
    report_path: Path,
    vocab_size: int,
    requested_vocab: int,
    metrics: dict,
    baseline: dict,
    examples: list[dict],
    min_frequency: int,
) -> None:
    L = [
        "# Hindi BPE tokenizer — report",
        "",
        "Generated by `scripts/train_tokenizer.py`.",
        "",
        "## Summary",
        "",
        f"- Requested vocab size: **{requested_vocab:,}**",
        f"- Actual vocab size: **{vocab_size:,}**  "
        + ("" if vocab_size >= requested_vocab
           else "(BPE stopped early — corpus too small to reach the target)"),
        f"- min_frequency: {min_frequency}",
        f"- Pipeline: NFC → Metaspace → BPE → Metaspace decode",
        "",
        "## Metrics (on a held-in sample)",
        "",
        f"- Sample lines: {metrics['sample_lines']:,}",
        f"- Sample words: {metrics['sample_words']:,}",
        f"- Sample tokens: {metrics['sample_tokens']:,}",
        f"- **Fertility** (tokens / word): "
        f"**{metrics['fertility_tokens_per_word']:.3f}**  "
        f"_(lower is better; Hindi words ideally 1.5–2.5)_",
        f"- **Compression** (chars / token): "
        f"**{metrics['compression_chars_per_token']:.3f}**  "
        f"_(higher is better)_",
        f"- **Unknown-token rate**: {metrics['unknown_token_rate']:.5f}  "
        f"_(should be ~0)_",
        "",
        "## Baseline comparison",
        "",
        f"- Baseline kind: `{baseline['kind']}`",
    ]
    if "fertility_tokens_per_word" in baseline:
        L += [
            f"- Baseline fertility: {baseline['fertility_tokens_per_word']:.3f}",
            f"- Baseline compression: {baseline['compression_chars_per_token']:.3f}",
        ]
    else:
        cl, bl = baseline["char_level"], baseline["byte_level_utf8"]
        L += [
            f"- char-level fertility: {cl['fertility_tokens_per_word']:.3f} "
            f"(compression {cl['compression_chars_per_token']:.3f})",
            f"- utf8-byte fertility: {bl['fertility_tokens_per_word']:.3f} "
            f"(compression {bl['compression_chars_per_token']:.3f})",
            "",
            "_These fallback baselines show how many units a naive char/byte "
            "scheme would emit per word. A good BPE should sit far below both._",
        ]
    L += [
        "",
        "## Tokenization examples",
        "",
    ]
    for ex in examples:
        L.append(f"- _{ex['text']}_")
        L.append(f"  - {ex['n_tokens']} tokens: `{' '.join(ex['tokens'])}`")
    L += [
        "",
        "## Why this vocab size?",
        "",
        "Vocab size trades off sequence length against parameter cost. A larger "
        "vocab lowers fertility (fewer tokens per Hindi word → shorter sequences "
        "→ more text per training step) but enlarges the embedding/output matrix "
        "(`vocab × d_model`) and risks under-trained rare tokens. ~32k balances "
        "short Hindi sequences against keeping the embedding a sensible share of "
        "a ~50M model. See `docs/tokenizer_notes.md`.",
        "",
    ]
    report_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Hindi BPE tokenizer and write a report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", nargs="+", default=["data/processed/clean.txt"],
                   help="Training corpus: .txt/.jsonl file(s) or directory.")
    p.add_argument("--output", default="tokenizer/hindi_bpe.json",
                   help="Path to write the tokenizer JSON.")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--text-field", default="text")
    p.add_argument("--sample-lines", type=int, default=5000,
                   help="Lines used to compute report metrics.")
    p.add_argument("--baseline-tokenizer", default=None,
                   help="Optional external tokenizer JSON for comparison.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    files = gather_files(args.input)

    print(f"Training BPE (vocab={args.vocab_size}) on {len(files)} file(s)...")
    tok = train_tokenizer(files, args.vocab_size, args.min_frequency, args.text_field)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    actual_vocab = tok.get_vocab_size()
    print(f"Saved tokenizer -> {out_path}  (vocab size = {actual_vocab})")

    # Build a metrics sample from the corpus.
    sample: list[str] = []
    for line in iter_lines(files, args.text_field):
        sample.append(line)
        if len(sample) >= args.sample_lines:
            break

    metrics = compute_metrics(tok, sample)
    baseline = baseline_metrics(sample, args.baseline_tokenizer)
    examples = tokenization_examples(tok, _DEFAULT_EXAMPLES)

    report_md = out_path.with_suffix(".report.md")
    report_json = out_path.with_suffix(".report.json")
    write_report(report_md, actual_vocab, args.vocab_size, metrics, baseline,
                 examples, args.min_frequency)
    report_json.write_text(
        json.dumps(
            {"vocab_size": actual_vocab, "requested_vocab": args.vocab_size,
             "metrics": metrics, "baseline": baseline},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== tokenizer report ===")
    print(f"vocab size       : {actual_vocab:,}")
    print(f"fertility (tok/w): {metrics['fertility_tokens_per_word']:.3f}")
    print(f"compression(c/t) : {metrics['compression_chars_per_token']:.3f}")
    print(f"unknown rate     : {metrics['unknown_token_rate']:.5f}")
    print(f"report           : {report_md}")
    if actual_vocab < args.vocab_size:
        print(f"\n[note] actual vocab ({actual_vocab}) < requested "
              f"({args.vocab_size}); the corpus was too small to reach the "
              f"target. This is expected on the tiny sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
