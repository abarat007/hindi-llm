"""hindi_llm: a small Hindi causal language model built from scratch in PyTorch.

The package is intentionally small and dependency-light. Public surface:

- ``config``         : the central configuration dataclasses + the 50M default.
- ``model``          : the GPT decoder (RoPE, RMSNorm, SwiGLU) built by hand.
- ``data``           : token-shard dataset + batch sampling.
- ``tokenizer_io``   : load/save helpers around the ``tokenizers`` library.
- ``train_utils``    : LR schedule, checkpointing, loss estimation, device setup.
- ``sampling``       : autoregressive generation (temperature / top-k / top-p).
- ``chat_template``  : the Hindi chat format used for SFT and inference.
- ``eval_utils``     : perplexity + qualitative generation helpers.
"""

__version__ = "0.1.0"
