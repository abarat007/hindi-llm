#!/usr/bin/env python3
"""Phase 1 — Hindi corpus cleaning, dedup, filtering, and stats.

Reads raw Hindi text you supply (``.txt`` / ``.jsonl`` / a directory of files),
cleans and filters it, removes exact and near duplicates, and writes:

  * ``clean.jsonl``      — one {"text": ...} per kept document
  * ``clean.txt``        — one kept document per line (for tokenizer training)
  * ``corpus_stats.json``— machine-readable counts and percentiles
  * ``data_card.md``     — human-readable data card with sample snippets

Why this matters (see docs/data_card_template.md for the long version): for a
low-resource language, *data quality dominates*. A 50M model has little capacity
to spare, so boilerplate, English contamination, and duplicates directly waste
it and encourage memorization. We therefore filter aggressively and report
exactly what each filter removed so the cleaning is auditable.

Deliberately dependency-light: standard library + numpy only. Devanagari is
detected by Unicode code-point range (no `regex` needed); near-dup uses a
hand-rolled MinHash + LSH so the algorithm is visible and defensible.

Example:
    python scripts/prepare_corpus.py \
        --input data/raw --output-dir data/processed \
        --txt-mode line --min-chars 100 --min-devanagari 0.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Character classification (Devanagari by Unicode block)
# --------------------------------------------------------------------------- #
# Devanagari        : U+0900–U+097F   (consonants, vowels, matras, digits ०-९)
# Devanagari Ext.   : U+A8E0–U+A8FF   (vedic / additional signs)
def _is_devanagari(ch: str) -> bool:
    o = ord(ch)
    return 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF


# A precompiled URL matcher (stdlib re is enough; no `regex` dependency).
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_WS_RE = re.compile(r"\s+")

# Conservative boilerplate line markers (case-insensitive substring match).
# Kept short on purpose; over-aggressive boilerplate stripping deletes real text.
_BOILERPLATE_MARKERS = (
    "all rights reserved",
    "सर्वाधिकार सुरक्षित",
    "copyright",
    "©",
    "cookie",
    "terms of service",
    "privacy policy",
    "read more",
    "click here",
    "advertisement",
    "विज्ञापन",
)


@dataclass
class CharStats:
    """Per-document character composition over *non-space* characters."""

    n_nonspace: int
    deva_ratio: float
    latin_ratio: float
    digit_ratio: float
    punct_ratio: float
    longest_run_ratio: float  # longest run of one repeated char / total length
    url_char_ratio: float     # fraction of characters inside URLs


def char_stats(text: str) -> CharStats:
    """Compute the ratios used by the quality filters.

    Each non-space char is bucketed into exactly one of {deva, digit, latin,
    other} with Devanagari taking priority (so Hindi numerals ०-९ count as
    Devanagari). Punctuation is counted separately via Unicode category.
    """
    deva = latin = digit = punct = nonspace = 0
    longest_run = 1 if text else 0
    run = 1
    prev = ""
    for ch in text:
        if ch == prev:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
        prev = ch
        if ch.isspace():
            continue
        nonspace += 1
        if unicodedata.category(ch).startswith("P"):
            punct += 1
        if _is_devanagari(ch):
            deva += 1
        elif ch.isascii() and ch.isdigit():
            digit += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif ch.isdigit():
            digit += 1
        # else: 'other' (symbols, emoji, other scripts) — falls through

    n = max(nonspace, 1)
    url_chars = sum(len(m.group(0)) for m in _URL_RE.finditer(text))
    return CharStats(
        n_nonspace=nonspace,
        deva_ratio=deva / n,
        latin_ratio=latin / n,
        digit_ratio=digit / n,
        punct_ratio=punct / n,
        longest_run_ratio=(longest_run / len(text)) if text else 0.0,
        url_char_ratio=(url_chars / len(text)) if text else 0.0,
    )


# --------------------------------------------------------------------------- #
# Normalization + boilerplate
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """NFC-normalize and tidy whitespace.

    NFC matters for Devanagari: the same visible akshara can be encoded with
    different code-point sequences (e.g. precomposed nukta vs base+nukta). NFC
    gives the tokenizer one canonical form, reducing spurious vocabulary.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")  # zero-width / BOM
    # normalize newlines, collapse runs of spaces/tabs but keep line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def strip_boilerplate(text: str) -> str:
    """Drop boilerplate-ish lines and collapse consecutive duplicate lines."""
    kept: list[str] = []
    last = None
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(mark in low for mark in _BOILERPLATE_MARKERS):
            continue
        # lines with no letters and no Devanagari (pure punctuation / symbols)
        if not any(ch.isalnum() or _is_devanagari(ch) for ch in s):
            continue
        if s == last:  # consecutive duplicate line
            continue
        kept.append(s)
        last = s
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
# Quality filter
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    min_chars: int = 100
    min_devanagari: float = 0.60
    max_latin: float = 0.20
    max_digit: float = 0.20
    max_punct: float = 0.30
    max_repeat_run: float = 0.20   # longest single-char run as fraction of length
    max_url: float = 0.10


# Ordered list of (reason, predicate) — first failing predicate attributes the
# removal. Attribution is "first failing filter", documented in the data card.
def quality_reason(text: str, cs: CharStats, th: Thresholds) -> str | None:
    """Return the name of the first filter the document fails, else None."""
    if len(text) < th.min_chars:
        return "too_short"
    if cs.deva_ratio < th.min_devanagari:
        return "low_devanagari"
    if cs.latin_ratio > th.max_latin:
        return "latin_heavy"
    if cs.digit_ratio > th.max_digit:
        return "digit_heavy"
    if cs.punct_ratio > th.max_punct:
        return "punct_heavy"
    if cs.longest_run_ratio > th.max_repeat_run:
        return "repeated_char"
    if cs.url_char_ratio > th.max_url:
        return "url_heavy"
    return None


# --------------------------------------------------------------------------- #
# Near-duplicate detection: MinHash + LSH banding
# --------------------------------------------------------------------------- #
class MinHashLSH:
    """Streaming near-dup detector over char-shingles.

    For each document we build a set of character 5-gram shingles, hash them to
    32-bit ints (crc32, stable across runs), and reduce to a MinHash signature
    of ``num_perm`` values via the universal family h(x) = (a*x + b) mod P.

    Two documents' signature agreement fraction estimates their Jaccard
    similarity. To avoid the O(n^2) all-pairs comparison we use LSH banding:
    the signature is split into ``bands`` bands; documents that collide in any
    band bucket are candidate duplicates and only those pairs are verified.
    """

    def __init__(
        self,
        num_perm: int = 64,
        bands: int = 16,
        shingle_k: int = 5,
        threshold: float = 0.8,
        seed: int = 1,
    ) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_k = shingle_k
        self.threshold = threshold

        # Keep a, b < 2**31 and x < 2**32 so a*x+b < 2**63 (no uint64 overflow
        # before the modulo). P is a Mersenne prime > 2**32.
        rng = np.random.default_rng(seed)
        self.P = np.uint64((1 << 61) - 1)
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

        self._buckets: dict[tuple, list[int]] = {}
        self._signatures: list[np.ndarray] = []

    def _shingle_hashes(self, text: str) -> np.ndarray:
        s = _WS_RE.sub(" ", text)
        k = self.shingle_k
        if len(s) <= k:
            grams = {s}
        else:
            grams = {s[i : i + k] for i in range(len(s) - k + 1)}
        # crc32 -> deterministic 32-bit hashes; [n_shingles]
        return np.fromiter(
            (zlib.crc32(g.encode("utf-8")) for g in grams),
            dtype=np.uint64,
            count=len(grams),
        )

    def _signature(self, text: str) -> np.ndarray:
        x = self._shingle_hashes(text)                       # [n_shingles]
        if x.size == 0:
            return np.full(self.num_perm, self.P, dtype=np.uint64)
        # (a[:,None]*x[None,:] + b[:,None]) % P  -> [num_perm, n_shingles]
        hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % self.P
        return hashed.min(axis=1)                            # [num_perm]

    def is_duplicate(self, text: str) -> bool:
        """Check against everything seen so far; if new, register and return False."""
        sig = self._signature(text)
        band_keys = []
        candidates: set[int] = set()
        for band in range(self.bands):
            rows = sig[band * self.rows : (band + 1) * self.rows]
            key = (band, rows.tobytes())                     # bucket id for band
            band_keys.append(key)
            candidates.update(self._buckets.get(key, ()))

        for idx in candidates:
            est_jaccard = float(np.mean(sig == self._signatures[idx]))
            if est_jaccard >= self.threshold:
                return True  # near-duplicate of an already-kept document

        # not a duplicate: register this document
        new_idx = len(self._signatures)
        self._signatures.append(sig)
        for key in band_keys:
            self._buckets.setdefault(key, []).append(new_idx)
        return False


# --------------------------------------------------------------------------- #
# Loading raw documents
# --------------------------------------------------------------------------- #
def _iter_file(path: Path, text_field: str, txt_mode: str) -> Iterator[str]:
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
    else:  # treat as plain text
        raw = path.read_text(encoding="utf-8", errors="replace")
        if txt_mode == "file":
            yield raw
        elif txt_mode == "paragraph":
            for para in re.split(r"\n\s*\n", raw):
                if para.strip():
                    yield para
        else:  # "line" (default)
            for line in raw.split("\n"):
                if line.strip():
                    yield line


def iter_raw_documents(
    inputs: list[Path], text_field: str, txt_mode: str
) -> Iterator[str]:
    """Yield raw document strings from files and/or directories."""
    files: list[Path] = []
    for inp in inputs:
        if inp.is_dir():
            files.extend(sorted(p for p in inp.rglob("*") if p.is_file()))
        elif inp.is_file():
            files.append(inp)
        else:
            print(f"[warn] input not found, skipping: {inp}", file=sys.stderr)
    for path in files:
        if path.suffix not in (".txt", ".jsonl"):
            continue
        yield from _iter_file(path, text_field, txt_mode)


# --------------------------------------------------------------------------- #
# Stats / data card
# --------------------------------------------------------------------------- #
def percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values)
    qs = [10, 25, 50, 75, 90, 99]
    out = {f"p{q}": float(np.percentile(arr, q)) for q in qs}
    out.update(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))
    return out


def write_data_card(
    out_path: Path,
    stats: dict,
    thresholds: Thresholds,
    samples: list[str],
) -> None:
    rem = stats["removed_by_filter"]
    lines = [
        "# Hindi corpus — generated data card",
        "",
        "> Auto-generated by `scripts/prepare_corpus.py`. This records *what the",
        "> cleaning did*. For provenance/licensing, fill in",
        "> `docs/data_card_template.md` by hand.",
        "",
        "## Document counts",
        "",
        f"- Raw documents read: **{stats['raw_docs']:,}**",
        f"- Documents kept: **{stats['kept_docs']:,}** "
        f"({stats['kept_fraction'] * 100:.1f}% of raw)",
        f"- Exact duplicates removed: **{stats['exact_duplicates']:,}**",
        f"- Near duplicates removed: **{stats['near_duplicates']:,}**",
        "",
        "## Removed by quality filter (first-failing-filter attribution)",
        "",
        "| Filter | Removed |",
        "| --- | ---: |",
    ]
    for name, count in rem.items():
        lines.append(f"| `{name}` | {count:,} |")
    lines += [
        "",
        "## Corpus size (kept documents)",
        "",
        f"- Total characters: **{stats['total_chars']:,}**",
        f"- Estimated words (whitespace): **{stats['est_words']:,}**",
        "",
        "## Document length (characters) percentiles",
        "",
        "| stat | value |",
        "| --- | ---: |",
    ]
    for k, v in stats["length_percentiles"].items():
        lines.append(f"| {k} | {v:,.0f} |")
    lines += [
        "",
        "## Devanagari ratio (kept documents)",
        "",
        f"- mean: {stats['devanagari_ratio']['mean']:.3f}",
        f"- median: {stats['devanagari_ratio']['median']:.3f}",
        f"- min: {stats['devanagari_ratio']['min']:.3f}",
        "",
        "## Filter thresholds used",
        "",
        "```json",
        json.dumps(thresholds.__dict__, indent=2),
        "```",
        "",
        "## Sample cleaned snippets (inspect these by hand!)",
        "",
    ]
    for i, snip in enumerate(samples, 1):
        preview = snip if len(snip) <= 300 else snip[:300] + " …"
        lines.append(f"{i}. {preview}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean, dedup, and filter a Hindi corpus; emit stats + data card.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", nargs="+", default=["data/raw"],
        help="Input file(s) and/or directory(ies): .txt / .jsonl / dirs.",
    )
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--text-field", default="text", help="JSONL text field name.")
    p.add_argument(
        "--txt-mode", choices=["line", "paragraph", "file"], default="line",
        help="How to split plain-text files into documents.",
    )
    # quality thresholds
    p.add_argument("--min-chars", type=int, default=100)
    p.add_argument("--min-devanagari", type=float, default=0.60)
    p.add_argument("--max-latin", type=float, default=0.20)
    p.add_argument("--max-digit", type=float, default=0.20)
    p.add_argument("--max-punct", type=float, default=0.30)
    p.add_argument("--max-repeat-run", type=float, default=0.20)
    p.add_argument("--max-url", type=float, default=0.10)
    # near-dup
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash near-dup detection (exact dedup still runs).")
    p.add_argument("--near-dup-threshold", type=float, default=0.80)
    p.add_argument("--num-perm", type=int, default=64)
    p.add_argument("--bands", type=int, default=16)
    p.add_argument("--shingle-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=10,
                   help="Cleaned snippets to include in the data card.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    th = Thresholds(
        min_chars=args.min_chars,
        min_devanagari=args.min_devanagari,
        max_latin=args.max_latin,
        max_digit=args.max_digit,
        max_punct=args.max_punct,
        max_repeat_run=args.max_repeat_run,
        max_url=args.max_url,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in args.input]

    removed = OrderedDict(
        empty=0, too_short=0, low_devanagari=0, latin_heavy=0, digit_heavy=0,
        punct_heavy=0, repeated_char=0, url_heavy=0,
    )
    exact_dups = 0
    near_dups = 0
    raw_docs = 0

    seen_hashes: set[str] = set()
    deduper = None if args.no_near_dedup else MinHashLSH(
        num_perm=args.num_perm, bands=args.bands, shingle_k=args.shingle_k,
        threshold=args.near_dup_threshold, seed=args.seed,
    )

    kept_lengths: list[int] = []
    kept_deva: list[float] = []
    total_chars = 0
    est_words = 0
    samples: list[str] = []

    clean_jsonl_path = out_dir / "clean.jsonl"
    clean_txt_path = out_dir / "clean.txt"

    with clean_jsonl_path.open("w", encoding="utf-8") as fj, \
            clean_txt_path.open("w", encoding="utf-8") as ft:
        for raw in tqdm(
            iter_raw_documents(inputs, args.text_field, args.txt_mode),
            desc="cleaning", unit="doc",
        ):
            raw_docs += 1
            text = strip_boilerplate(normalize_text(raw))
            if not text:
                removed["empty"] += 1
                continue

            cs = char_stats(text)
            reason = quality_reason(text, cs, th)
            if reason is not None:
                removed[reason] += 1
                continue

            # exact dedup on the normalized text
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                exact_dups += 1
                continue
            seen_hashes.add(h)

            # near dedup
            if deduper is not None and deduper.is_duplicate(text):
                near_dups += 1
                continue

            # keep it: write one doc per line (newlines -> spaces for the .txt)
            fj.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            ft.write(text.replace("\n", " ") + "\n")

            kept_lengths.append(len(text))
            kept_deva.append(cs.deva_ratio)
            total_chars += len(text)
            est_words += len(text.split())
            if len(samples) < args.num_samples:
                samples.append(text.replace("\n", " "))

    kept_docs = len(kept_lengths)
    deva_arr = np.asarray(kept_deva) if kept_deva else np.asarray([0.0])
    stats = {
        "raw_docs": raw_docs,
        "kept_docs": kept_docs,
        "kept_fraction": (kept_docs / raw_docs) if raw_docs else 0.0,
        "removed_by_filter": dict(removed),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "total_chars": total_chars,
        "est_words": est_words,
        "length_percentiles": percentiles(kept_lengths),
        "devanagari_ratio": {
            "mean": float(deva_arr.mean()),
            "median": float(np.median(deva_arr)),
            "min": float(deva_arr.min()),
        },
        "thresholds": th.__dict__,
        "near_dedup_enabled": deduper is not None,
    }

    stats_path = out_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), "utf-8")
    write_data_card(out_dir / "data_card.md", stats, th, samples)

    # console summary
    print("\n=== corpus preparation summary ===")
    print(f"raw documents      : {raw_docs:,}")
    print(f"kept documents     : {kept_docs:,} "
          f"({stats['kept_fraction'] * 100:.1f}%)")
    print(f"exact duplicates   : {exact_dups:,}")
    print(f"near duplicates    : {near_dups:,}")
    for name, count in removed.items():
        if count:
            print(f"removed [{name:<14}]: {count:,}")
    print(f"total characters   : {total_chars:,}")
    print(f"estimated words    : {est_words:,}")
    print(f"\nwrote: {clean_jsonl_path}")
    print(f"wrote: {clean_txt_path}")
    print(f"wrote: {stats_path}")
    print(f"wrote: {out_dir / 'data_card.md'}")
    if kept_docs == 0:
        print("\n[warn] no documents kept — loosen thresholds or check --input.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
