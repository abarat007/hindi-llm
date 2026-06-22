# Data card (template — fill this in before training)

`scripts/prepare_corpus.py` auto-generates a *statistics* card
(`data/processed/data_card.md`) describing what the cleaning did. **This file is
different**: it records provenance, licensing, and known biases — things only you
know about your sources. Fill it in before you train, and keep it with the model.

---

## 1. Corpus sources

List every source, with version/date and how much you took.

| Source | Version / snapshot date | Approx. size (docs / tokens) | Notes |
|---|---|---|---|
| Hindi Wikipedia | TODO (dump date) | TODO | encyclopedic, clean |
| OSCAR (hi) | TODO | TODO | web crawl, noisy |
| Common Crawl (hi slices) | TODO | TODO | very noisy |
| IndicCorp-style Hindi | TODO | TODO | check license |
| (your own) | TODO | TODO | TODO |

## 2. License / usage rights

For each source, record the license and whether your intended use is permitted.

- Hindi Wikipedia: CC BY-SA — TODO confirm attribution/share-alike obligations.
- OSCAR: per-document licensing is mixed; TODO record the OSCAR release terms.
- Common Crawl: TODO — respect robots/source terms.
- IndicCorp / others: TODO — confirm research vs commercial use.

> If you cannot establish rights to a source, do not include it.

## 3. Cleaning filters applied

These are produced by `prepare_corpus.py`; record the thresholds you used (they
are also saved in `corpus_stats.json`).

- Unicode NFC normalization, whitespace tidy, zero-width/BOM removal.
- Boilerplate line removal (cookie/copyright/nav markers, pure-punctuation lines,
  consecutive duplicates).
- Quality filters (first-failing-filter attribution):
  - `min_chars` = TODO
  - `min_devanagari` (Devanagari ratio) = TODO
  - `max_latin`, `max_digit`, `max_punct` = TODO
  - `max_repeat_run` (repeated-char run) = TODO
  - `max_url` (URL-heavy) = TODO
- Exact dedup (SHA-1) and near-dedup (MinHash+LSH, threshold = TODO).

Paste the final counts from `data_card.md` here (raw vs kept, removed-by-filter,
exact/near duplicates) so the record is self-contained.

## 4. Known biases

Be explicit. Web Hindi corpora skew toward:

- **Topic/domain**: news, religion, entertainment, government — under-represents
  conversational, technical, and regional content.
- **Register/dialect**: Standard (Khari Boli) Hindi dominates; dialects
  (Bhojpuri, Awadhi, etc.) and code-mixed Hinglish are under- or mis-represented.
- **Demographic/source skew**: whoever publishes Hindi web text. Document any
  obvious gaps.
- **Quality artifacts**: machine-translated text, SEO spam, transliteration
  noise. Note if you could not fully filter these.

## 5. Hindi coverage

- Script: Devanagari (with NFC). Note any transliterated/Latin Hindi you kept or
  dropped.
- Numerals: Devanagari (०–९) vs ASCII — note which dominates after filtering.
- Approx. Devanagari ratio of the kept corpus (from `data_card.md`): TODO.

## 6. Failure cases / things to inspect by hand

Before tokenizing, *read* a few hundred random cleaned snippets and check for:

- residual boilerplate or navigation text,
- English/Hinglish that slipped past the Latin filter,
- repeated/templated documents that dedup missed,
- broken Devanagari (mojibake, wrong normalization),
- offensive or sensitive content you don't want memorized.

Record what you found and any extra filtering you applied.

## 7. Reproducibility

- Exact `prepare_corpus.py` command(s) and thresholds: TODO.
- Tokenizer training command + vocab size: TODO.
- Encoding command + val fraction: TODO.
- Commit hash of this repo at training time: TODO.
