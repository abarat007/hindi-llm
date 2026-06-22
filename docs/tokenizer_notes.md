# Tokenizer notes

The tokenizer is trained by [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py)
and wrapped by [`src/hindi_llm/tokenizer_io.py`](../src/hindi_llm/tokenizer_io.py).

## Pipeline

```
NFC normalize  →  Metaspace pre-tokenize (▁)  →  BPE merges  →  Metaspace decode
```

- **NFC normalization** canonicalizes Devanagari. The same visible *akshara* can
  be encoded by different code-point sequences (e.g. a precomposed nukta letter
  vs base letter + combining nukta U+093C). Without normalization these fork into
  separate tokens, wasting vocab and fragmenting frequencies. NFC collapses them.
- **Metaspace** marks word boundaries with `▁` (the SentencePiece convention).
  Hindi is space-separated, so this is a natural, fully reversible boundary
  scheme. Decoding turns `▁` back into spaces.
- **BPE** greedily merges frequent adjacent pieces. Starting from characters, it
  learns subwords up to the target vocab size.

## What is BPE, briefly

Byte-Pair Encoding starts with a base alphabet (here: the Devanagari characters
seen in the corpus, plus some Latin/punct/digits) and repeatedly merges the most
frequent adjacent pair into a new symbol until it reaches the vocab size. Frequent
whole words become single tokens; rare words decompose into subword pieces. This
gives an open vocabulary (any string is representable) with a fixed-size table.

## Special tokens

Seven, fixed in `config.SPECIAL_TOKENS`: `<pad>`, `<bos>`, `<eos>`, `<unk>`,
`<system>`, `<user>`, `<assistant>`. The chat tokens are reserved at tokenizer
training time so the chat template can address them by id (see
[`chat_template.py`](../src/hindi_llm/chat_template.py)).

## The metrics the report tracks

Run produces a `*.report.md` next to the tokenizer. Key numbers:

- **Fertility = tokens / whitespace-word.** How many tokens, on average, a Hindi
  word costs. Lower is better. A good Hindi BPE lands around ~1.5–2.5 on real
  text. (On the tiny tracked sample it can read ~1.0 because BPE memorizes whole
  words — not representative.)
- **Compression = characters / token.** Higher means each token carries more
  text. The inverse view of fertility.
- **Unknown-token rate.** Fraction of tokens that are `<unk>`. Should be ~0 on
  in-domain Hindi because the base alphabet is fully learned. We *measure and
  report* it rather than assume it.

## Baseline comparison

The report compares against a baseline so the numbers mean something:

- If you pass `--baseline-tokenizer path/to/tokenizer.json` (e.g. a downloaded
  GPT-2 tokenizer), it is used directly.
- Otherwise the script emits **clearly-labeled fallback estimates**: a
  char-level scheme (tokens = characters) and a UTF-8 byte-level scheme
  (tokens = bytes). We never pretend a real external baseline exists. The point
  of the fallbacks: a Devanagari character is ~3 UTF-8 bytes, so a naive
  byte-level scheme has very high fertility for Hindi; a good BPE should sit far
  below it.

## Why fertility matters (especially for Hindi / low-resource)

Sequence length is *fertility × words*. Training compute and memory scale with
sequence length (attention is `O(T²)` in the sequence dim, and you simply have to
process more tokens). A tokenizer that needs 3 tokens per Hindi word instead of
1.8 makes every sequence ~65% longer — you see less text per step, pay more per
step, and the effective context (in words) shrinks. English-centric tokenizers
(like vanilla GPT-2's byte-level BPE) are notoriously *fertile* on Devanagari,
which is a major reason a **Hindi-specific tokenizer** is worth training.

## Vocab-size tradeoff

`vocab_size` is both a data and an **architecture** decision (it sets the
embedding/output matrix size `vocab × d_model`).

- **Too small** (e.g. 8k): high fertility, long Hindi sequences, more compute,
  shorter effective context. The model spends capacity reassembling words from
  fragments.
- **Too large** (e.g. 100k for a 50M model): the `vocab × d_model` embedding
  dominates the parameter budget, and many rare tokens are seen too few times to
  train well (under-trained embeddings → noise). It also overfits rare fragments.
- **~32k** balances short Hindi sequences against keeping the embedding a sensible
  share (~16M of ~48M) of a 50M model. If you scale the model up (see
  [`scaling_10x.md`](scaling_10x.md)) a larger vocab becomes affordable.

## Hindi-specific issues to watch

- **Matras / combining marks**: dependent vowel signs and the halant (्) attach
  to consonants. NFC + character-level BPE handles these, but rare conjuncts can
  fragment into many pieces (local fertility spikes) and show up as spelling slips
  in generation.
- **Mixed scripts / numerals**: Devanagari digits (०–९) vs ASCII digits, and
  English loanwords in Latin script. The corpus cleaner's Latin/digit ratio
  filters keep contamination low so the tokenizer doesn't waste merges on English.
- **Inconsistent encodings** across sources — handled by NFC, but worth spot-
  checking the tokenization examples in the report.
