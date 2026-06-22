"""Thin, explicit wrapper around the `tokenizers` library.

We use the low-level `tokenizers` API directly (no HuggingFace Trainer, no
`AutoTokenizer`). This module centralizes:

  * loading a trained tokenizer JSON,
  * looking up the special-token ids we rely on (<bos>, <eos>, <pad>, ...),
  * encode / decode helpers used across training, SFT, and generation.

Keeping all tokenizer access behind one small class means the rest of the code
never touches `tokenizers` internals and the special-token contract is defined
in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from .config import SPECIAL_TOKENS


class HindiTokenizer:
    """Wraps a trained `tokenizers.Tokenizer` with our special-token contract."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        # Resolve special-token ids once; fail loudly if any is missing so we
        # never silently train/generate with a mis-built tokenizer.
        self.special_ids: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            tid = self._tok.token_to_id(tok)
            if tid is None:
                raise ValueError(
                    f"Special token {tok!r} not found in tokenizer vocab. "
                    f"Retrain with scripts/train_tokenizer.py."
                )
            self.special_ids[tok] = tid

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "HindiTokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. Train one first:\n"
                f"  python scripts/train_tokenizer.py --input <corpus> "
                f"--output {path}"
            )
        return cls(Tokenizer.from_file(str(path)))

    # -- convenience ids -----------------------------------------------------
    @property
    def bos_id(self) -> int:
        return self.special_ids["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_ids["<eos>"]

    @property
    def pad_id(self) -> int:
        return self.special_ids["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.special_ids["<unk>"]

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    # -- encode / decode -----------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to ids. BOS/EOS are added here (not via post-processors)
        so the behavior is explicit at every call site."""
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def __len__(self) -> int:
        return self.vocab_size
