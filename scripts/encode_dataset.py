#!/usr/bin/env python3
"""Encode a cleaned Hindi corpus into flat token shards for training.

Reads the cleaned corpus (``clean.jsonl`` or ``clean.txt``), tokenizes each
document with the trained tokenizer, appends an ``<eos>`` after every document
(so the model learns document boundaries), concatenates everything into one
flat token stream, splits off a validation tail, and writes:

  * ``train.bin`` / ``val.bin``  — raw little-endian token ids (uint16 or uint32)
  * ``meta.json``                — dtype, vocab size, token counts, eos id

The ``.bin`` format is deliberately trivial so the data loader can ``np.memmap``
it without any parsing. Memory use during encoding is ~2 bytes/token via the
stdlib ``array``; the train/val split is written as zero-copy views.

Example:
    python scripts/encode_dataset.py \
        --input data/processed/clean.jsonl \
        --tokenizer tokenizer/hindi_bpe.json \
        --output-dir data/processed --val-fraction 0.0005
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hindi_llm.data import dtype_for_vocab  # noqa: E402
from hindi_llm.tokenizer_io import HindiTokenizer  # noqa: E402


def iter_documents(inputs: list[Path], text_field: str) -> Iterator[str]:
    for path in inputs:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and text_field in obj:
                        yield str(obj[text_field])
        else:  # .txt -> one document per line
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n")


def gather_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix in (".txt", ".jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"[warn] input not found: {p}", file=sys.stderr)
    if not out:
        raise FileNotFoundError(f"No .txt/.jsonl inputs found in: {inputs}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tokenize a cleaned corpus into train/val .bin token shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", nargs="+", default=["data/processed/clean.jsonl"])
    p.add_argument("--tokenizer", default="tokenizer/hindi_bpe.json")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--text-field", default="text")
    p.add_argument("--val-fraction", type=float, default=0.0005,
                   help="Fraction of tokens (from the tail) held out for validation.")
    p.add_argument("--add-bos", action="store_true",
                   help="Prepend <bos> before each document as well as <eos> after.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = gather_inputs(args.input)
    tok = HindiTokenizer.load(args.tokenizer)

    vocab_size = tok.vocab_size
    dtype = dtype_for_vocab(vocab_size)
    # 'H' = unsigned short (uint16), 'I' = unsigned int (uint32)
    typecode = "H" if dtype == np.uint16 else "I"
    buf = array(typecode)

    n_docs = 0
    for doc in tqdm(iter_documents(inputs, args.text_field), desc="encoding", unit="doc"):
        ids = tok.encode(doc, add_bos=args.add_bos, add_eos=True)
        buf.extend(ids)
        n_docs += 1

    total = len(buf)
    if total == 0:
        print("[error] no tokens produced — check --input/--tokenizer.", file=sys.stderr)
        return 1

    # zero-copy view over the array's buffer, then split off a validation tail
    all_ids = np.frombuffer(buf, dtype=dtype)            # [N]
    n_val = int(total * args.val_fraction)
    # always keep at least a few hundred val tokens if any val is requested
    if args.val_fraction > 0:
        n_val = max(n_val, min(256, total // 10))
    n_train = total - n_val
    train_ids = all_ids[:n_train]                        # [n_train]
    val_ids = all_ids[n_train:]                          # [n_val]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": vocab_size,
        "n_docs": n_docs,
        "total_tokens": int(total),
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "eos_id": tok.eos_id,
        "bos_id": tok.bos_id,
        "tokenizer": str(args.tokenizer),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")

    print("\n=== encoding summary ===")
    print(f"documents        : {n_docs:,}")
    print(f"total tokens     : {total:,}")
    print(f"train tokens     : {n_train:,}  -> {train_path}")
    print(f"val tokens       : {n_val:,}  -> {val_path}")
    print(f"dtype            : {meta['dtype']} (vocab={vocab_size})")
    print(f"meta             : {meta_path}")
    if n_val == 0:
        print("[warn] validation split is empty; raise --val-fraction.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
