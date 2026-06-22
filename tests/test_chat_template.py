"""Tests for the Hindi chat template and its loss masking."""

from __future__ import annotations

from pathlib import Path

import train_tokenizer  # from scripts/, via conftest path

from hindi_llm.chat_template import (
    DEFAULT_SYSTEM,
    ROLE_TOKENS,
    Turn,
    build_chat_ids,
    normalize_messages,
    render_text,
)
from hindi_llm.tokenizer_io import HindiTokenizer


def _tok(sample_corpus: Path, tmp_path: Path) -> HindiTokenizer:
    t = train_tokenizer.train_tokenizer(
        files=[sample_corpus], vocab_size=500, min_frequency=1, text_field="text"
    )
    out = tmp_path / "tok.json"
    t.save(str(out))
    return HindiTokenizer.load(out)


def test_default_system_prepended():
    turns = normalize_messages([Turn("user", "नमस्ते")])
    assert turns[0].role == "system"
    assert turns[0].content == DEFAULT_SYSTEM


def test_structure_bos_and_roles(sample_corpus, tmp_path):
    tok = _tok(sample_corpus, tmp_path)
    msgs = [Turn("user", "भारत क्या है?"), Turn("assistant", "भारत एक देश है।")]
    ids, mask = build_chat_ids(tok, msgs)
    assert ids[0] == tok.bos_id
    assert len(ids) == len(mask)
    # all three role tokens should appear
    for role in ("system", "user", "assistant"):
        assert tok.token_to_id(ROLE_TOKENS[role]) in ids


def test_only_assistant_tokens_are_trained(sample_corpus, tmp_path):
    tok = _tok(sample_corpus, tmp_path)
    msgs = [Turn("user", "भारत क्या है?"), Turn("assistant", "भारत एक देश है।")]
    ids, mask = build_chat_ids(tok, msgs)

    assistant_id = tok.token_to_id(ROLE_TOKENS["assistant"])
    # some tokens are trained, but not the majority (prompt is masked)
    assert any(mask)
    assert not all(mask)
    # the assistant *role marker* must not be trained on
    for tid, m in zip(ids, mask):
        if tid == assistant_id:
            assert m is False
    # the final trained token must be eos (model learns to stop)
    trained_ids = [tid for tid, m in zip(ids, mask) if m]
    assert trained_ids[-1] == tok.eos_id


def test_generation_prompt_ends_with_assistant(sample_corpus, tmp_path):
    tok = _tok(sample_corpus, tmp_path)
    ids, mask = build_chat_ids(tok, [Turn("user", "नमस्ते")],
                               add_generation_prompt=True)
    assert ids[-1] == tok.token_to_id(ROLE_TOKENS["assistant"])
    assert mask[-1] is False  # nothing to train in a generation prompt


def test_render_text_has_tags():
    text = render_text([Turn("user", "नमस्ते")])
    assert "<system>" in text and "</system>" in text
    assert "<user>" in text and "</user>" in text
