"""Token-shard dataset and batch sampling.

After encoding, the corpus is a single flat stream of token ids saved as a
binary file (``train.bin`` / ``val.bin``). At train time we memory-map the file
(so it never has to fit in RAM) and sample random windows of ``block_size``
contiguous tokens. The target is the input shifted by one — standard next-token
prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """uint16 is enough for vocab <= 65536; otherwise uint32."""
    return np.dtype(np.uint16) if vocab_size <= (1 << 16) else np.dtype(np.uint32)


def load_meta(meta_path: str | Path) -> dict:
    with Path(meta_path).open("r", encoding="utf-8") as f:
        return json.load(f)


class BinDataset:
    """Memory-mapped flat token stream with random-window batch sampling."""

    def __init__(self, path: str | Path, dtype: np.dtype = np.uint16) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Token shard not found: {path}. Encode the corpus first:\n"
                f"  python scripts/encode_dataset.py --input <clean.jsonl> "
                f"--tokenizer <tok.json>"
            )
        # mmap mode 'r' -> the OS pages tokens in on demand; no full load.
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of (x, y) windows.

        x[b] = tokens[i : i+block_size]
        y[b] = tokens[i+1 : i+1+block_size]   (next-token targets)
        """
        max_start = len(self.data) - block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset too small ({len(self.data)} tokens) for block_size "
                f"{block_size}. Use a smaller block_size or more data."
            )
        # random start offsets -> [B]
        ix = torch.randint(max_start, (batch_size,), generator=generator)

        # gather windows; cast to int64 (embedding indices must be long)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + block_size].astype(np.int64)) for i in ix]
        )  # [B, T]
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )  # [B, T]

        if device.startswith("cuda"):
            # pinned memory enables async host->device copy
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
