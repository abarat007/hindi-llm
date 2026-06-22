# `data/` directory

This directory holds corpora at various processing stages. **Large data is never
committed** (see the repo `.gitignore`); only directory placeholders and a tiny
sample are tracked.

## Layout

```text
data/
├── raw/         # original corpus files you supply (.txt / .jsonl / dirs)  [gitignored]
├── interim/     # intermediate artifacts during cleaning                  [gitignored]
├── processed/   # cleaned corpus + encoded token shards                   [gitignored]
├── sample_hindi.txt   # tiny tracked sample for smoke tests (NOT for real training)
└── sample_sft.jsonl   # tiny tracked instruction sample for SFT smoke tests
```

## Pipeline stages

1. **raw** — You drop your source files here. Supported inputs:
   - a single `.txt` file (one document per line, or paragraph-separated),
   - a `.jsonl` file with a configurable text field (default `text`),
   - a directory containing any mix of the above.
2. **interim/processed** — `scripts/prepare_corpus.py` reads `raw/`, cleans and
   deduplicates, and writes a cleaned `.jsonl` + plain `.txt` + a stats/data card.
3. **processed** — `scripts/encode_dataset.py` tokenizes the cleaned text into
   flat `uint16`/`uint32` token shards (`train.bin` / `val.bin`) for training.

## About the sample files

- `sample_hindi.txt` — ~50 short Hindi paragraphs. This exists **only** so the
  full pipeline, tokenizer training, and unit tests can run end-to-end in seconds
  on a laptop CPU. **It is far too small to train a usable language model.**
  A real run needs on the order of hundreds of millions to billions of Hindi
  tokens (Hindi Wikipedia, the Hindi subset of OSCAR, Hindi Common Crawl slices,
  IndicCorp-style text, etc.).
- `sample_sft.jsonl` — 12 toy instruction/response pairs in the project's chat
  schema, used to smoke-test the SFT path.

## Where to get real Hindi data

You supply this yourself (we deliberately do not scrape). Common sources:

| Source                  | Notes                                                        |
|-------------------------|-------------------------------------------------------------|
| Hindi Wikipedia         | Clean, encyclopedic; small but high quality.                |
| OSCAR (Hindi subset)    | Large web crawl; needs aggressive cleaning/dedup.           |
| Common Crawl (hi slices)| Very large, very noisy; quality filtering is essential.     |
| IndicCorp-style Hindi   | Curated Indic corpora; check licensing before use.          |

Always record provenance and licensing in `docs/data_card_template.md` before
training, and inspect cleaned samples by hand (see `docs/training_notes.md`).
