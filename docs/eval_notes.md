# Evaluation notes

Evaluation is intentionally modest and **honest**. A ~50M model on a modest Hindi
corpus is a research toy; the goal is to measure it fairly, not to oversell it.
Driven by [`scripts/evaluate.py`](../scripts/evaluate.py) and
[`src/hindi_llm/eval_utils.py`](../src/hindi_llm/eval_utils.py).

## Quantitative: validation perplexity

`eval_utils.corpus_perplexity` sweeps the **val shard** in non-overlapping
windows (deterministic, unlike the random-batch estimate used during training)
and reports `exp(mean NLL)`.

- Lower is better.
- Comparable **only** for a fixed tokenizer/vocab. Do not compare our perplexity
  to a model with a different tokenizer — the units differ.
- Use it to compare *our* runs (more data, more steps, different LR) against each
  other, and to confirm the model is learning Hindi structure.

## Qualitative: generations

Numbers don't tell you if the Hindi is *good*. So we also generate:

- **Base model**: free-form continuations of neutral Hindi prompts
  (`eval.prompts` in the config) — tests raw language modeling.
- **SFT model**: chat responses to a fixed Hindi prompt set (`CHAT_PROMPTS` in
  `evaluate.py`) using the chat template — tests instruction following.

Both are written to a Markdown report (`outputs/eval_report.md`) with the prompts
and outputs side by side, so quality is auditable by a human reader.

## The Hindi prompt set

Kept small, neutral, and concrete so failure modes are obvious:

- factual: "भारत की राजधानी क्या है?", "सूरज किस दिशा से उगता है?"
- explanatory: "पानी की तीन अवस्थाएँ कौन-सी हैं?"
- open continuation: "एक बार की बात है, एक छोटे से गाँव में"

Extend these to match your corpus's domain. Keep a *fixed* set across runs so
qualitative comparisons are meaningful.

## Sampling settings used

`temperature`, `top_k`, `top_p` come from the config's `eval` block (defaults
0.8 / 40 / 0.95). Lower temperature → more deterministic/repetitive; higher →
more diverse but more incoherent. For a weak model, top-p around 0.9–0.95 with
temperature ~0.7–0.8 usually reads best.

## Failure modes (documented, not hidden)

The report appends explicit failure-mode notes (also in `evaluate.py`):
repetition/loops, English code-switching, factual errors/hallucination, weak
stopping (run-on/truncation), and Devanagari spelling slips on rare conjuncts.
These are expected at this scale and shrink with more/cleaner data, more
parameters, and more SFT examples.

## What we deliberately do NOT claim

- No standardized benchmark scores (we don't run, e.g., IndicGLUE) — out of scope
  for a from-scratch toy, and easy to misreport.
- No comparison to large production Hindi models — different league.
- No metric appears in the README unless a script actually produced it; placeholders
  are clearly marked `TODO` until you run the real training.

## Future evaluation (if scaling up)

See [`scaling_10x.md`](scaling_10x.md). At larger scale, add: held-out perplexity
by domain, a small human-rated Hindi instruction set, and targeted probes
(grammar, factual recall, refusal behavior).
