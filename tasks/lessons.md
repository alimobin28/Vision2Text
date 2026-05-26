# Lessons Learned

A running log of bugs caught, decisions made, and things worth remembering.

---

## Bug 1 — `str.replace()` silently ignores regex patterns

**What happened:** The original notebook had lines like:
```python
caption = caption.str.replace("[^a-zA-Z]", "")
caption = caption.str.replace("'s", "")
```
`str.replace` treats the first argument as a *literal string*, not a pattern.
`"[^a-zA-Z]"` was never matched because there is no literal character sequence `[^a-zA-Z]`
in any caption. Every caption came out of "cleaning" completely unchanged.

**Fix:** Use `re.sub` with compiled patterns:
```python
_PUNCTUATION = re.compile(r"[^\w\s]")
caption = _PUNCTUATION.sub("", caption)
```

**Why it matters:** This is a completely silent failure — no error, no warning, wrong output.
All downstream vocab and model training was built on dirty captions.

---

## Bug 2 — Hardcoded Kaggle dataset paths

**What happened:** Paths like `/kaggle/input/flickr8k/Images/` and
`/kaggle/input/flickr8k/captions.txt` were copy-pasted across 8+ cells. Running the
notebook locally or on a different machine required manually hunting down every occurrence.

**Fix:** Every path lives in `src/config.py` as a `Path` field. Override via `.env` or
environment variable. Nothing in `src/` or `tests/` should ever contain a hardcoded path.

---

## Bug 3 — Feature extraction ran on every notebook restart

**What happened:** The EfficientNet forward pass over all 8,000 images ran every time the
notebook kernel was restarted. With no GPU warmup this took 3–5 minutes per session.

**Fix:** `src/features/extractor.py` checks for a `.h5` file whose name is a 12-char MD5
hash of the cache-relevant config (backbone name, image size, data directory). If the file
exists, extraction is skipped entirely. Changing the backbone or image size automatically
invalidates the cache and triggers a fresh extraction.

---

## Decision — HuggingFace `tokenizers` over `nltk` word-level vocab

Using `tokenizers.BPE` instead of a simple word-frequency vocab because:
- Handles out-of-vocabulary words gracefully (subword units)
- The same tokenizer interface works if we later swap to a pretrained LM decoder
- `vocab.json` is a self-contained, portable artifact — no NLTK data downloads needed

---

## Decision — Pydantic v2 `BaseSettings` over `argparse` / `yaml`

- Type validation is free (wrong path type raises immediately, not at training time)
- `.env` override means no code changes for different environments (local / Colab / cloud)
- `cfg.model_dump()` gives a serialisable snapshot of every run's config for logging
