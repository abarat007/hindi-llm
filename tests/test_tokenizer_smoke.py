"""Smoke test: train a tiny BPE tokenizer and check the basic contract.

Trains on the tracked sample corpus into a tmp dir (so it never touches the
gitignored ``tokenizer/`` artifacts), then verifies special tokens, encode/decode
roundtrip, and that the report metrics are sane.
"""

from __future__ import annotations

from pathlib import Path

import train_tokenizer  # from scripts/, added to path by conftest

from hindi_llm.config import SPECIAL_TOKENS
from hindi_llm.tokenizer_io import HindiTokenizer


def _train_tiny(sample_corpus: Path, out: Path) -> HindiTokenizer:
    tok = train_tokenizer.train_tokenizer(
        files=[sample_corpus], vocab_size=500, min_frequency=1, text_field="text"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return HindiTokenizer.load(out)


def test_special_tokens_present(sample_corpus, tmp_path):
    t = _train_tiny(sample_corpus, tmp_path / "tok.json")
    for sp in SPECIAL_TOKENS:
        assert t.token_to_id(sp) is not None, f"missing special token {sp}"
    # the convenience ids must be distinct
    ids = {t.bos_id, t.eos_id, t.pad_id, t.unk_id}
    assert len(ids) == 4


def test_encode_decode_roundtrip(sample_corpus, tmp_path):
    t = _train_tiny(sample_corpus, tmp_path / "tok.json")
    text = "भारत एक विशाल देश है।"
    ids = t.encode(text)
    assert len(ids) > 0
    assert t.decode(ids).strip() == text


def test_bos_eos_wrapping(sample_corpus, tmp_path):
    t = _train_tiny(sample_corpus, tmp_path / "tok.json")
    ids = t.encode("नमस्ते दुनिया।", add_bos=True, add_eos=True)
    assert ids[0] == t.bos_id
    assert ids[-1] == t.eos_id


def test_fertility_and_unknown_rate(sample_corpus, tmp_path):
    t = _train_tiny(sample_corpus, tmp_path / "tok.json")
    sample = sample_corpus.read_text(encoding="utf-8").splitlines()
    metrics = train_tokenizer.compute_metrics(t._tok, sample)
    # fertility should be a positive, finite number of tokens per word
    assert metrics["fertility_tokens_per_word"] > 0
    # on in-domain text the unknown rate must be ~0
    assert metrics["unknown_token_rate"] < 0.01
