# hindi-llm

A small **Hindi causal language model trained from scratch in PyTorch**, nanoGPT-style,
then fine-tuned into a minimal Hindi chat model.

> Status: scaffolding in place. Full case-study README, training curves, and
> qualitative samples are added as the project progresses.

The goal is an end-to-end, **readable and interview-defensible** pipeline:
data cleaning → Hindi BPE tokenizer → ~50M-parameter GPT (RoPE + RMSNorm + SwiGLU)
→ pretraining loop → Hindi SFT → evaluation → Gradio demo.

No `HuggingFace Trainer`, no `nn.Transformer`, no prebuilt model classes — every
layer and the training loop are written by hand with shape comments throughout.

See `docs/` for design notes and tradeoffs. Quickstart commands and the full
write-up land in later commits.
