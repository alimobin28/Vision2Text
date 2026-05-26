# Vision2Text — Project Scaffold, Config & Data Pipeline

## Context

Starting from an empty repo. The original work was a monolithic Kaggle notebook with three
hard bugs that silently corrupted every downstream experiment:

1. **Silent regex bug** — `str.replace("pattern", "")` does nothing when the "pattern"
   looks like a regex. Captions were never actually cleaned. Fixed with `re.sub`.
2. **Hardcoded Kaggle paths** — scattered across ~8 cells. Impossible to re-run elsewhere.
   Fixed by centralising every path in `src/config.py`.
3. **Feature extraction repeated every run** — no caching; EfficientNet ran on every notebook
   restart, wasting GPU time. Fixed with content-addressed `.h5` cache in `src/features/`.
4. **No structure** — everything in one file, untestable, unreusable.

---

## Directory Layout

```
Vision2Text/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic BaseSettings — single source of truth
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # torch Dataset wrapping Flickr captions + image paths
│   │   ├── preprocessing.py       # re.sub-based caption cleaning pipeline
│   │   └── vocab.py               # BPE tokenizer (HuggingFace tokenizers); saves vocab.json
│   └── features/
│       ├── __init__.py
│       └── extractor.py           # Pluggable CNN backbone; content-addressed .h5 cache
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_vocab.py
├── tasks/
│   ├── plan.md                    # this file
│   └── lessons.md
├── requirements.txt
└── .gitignore
```

---

## Module Responsibilities

### `src/config.py`
Single `Config(BaseSettings)` class. Every hyperparameter, path, and seed lives here.
Override any value via `.env` file or environment variable — no code changes needed.

### `src/data/preprocessing.py`
`clean_caption(text)` pipeline using compiled `re.compile` patterns:
- Lowercase → strip possessives → remove punctuation → drop pure numbers → collapse whitespace

### `src/data/vocab.py`
BPE tokenizer trained from scratch on cleaned captions via `tokenizers` library.
Saves to `vocab.json`; exposes `build_vocab`, `load_vocab`, `encode`.

### `src/data/dataset.py`
`FlickrDataset` — torch Dataset over Flickr8k/30k. Cleans captions on load,
returns `(image_tensor, caption_ids_tensor)` pairs.

### `src/features/extractor.py`
`extract_features(cfg, dataset)` — checks for a cached `.h5` file keyed by
`md5(backbone + image_size + data_dir)`. Skips extraction if cache exists.

---

## Verification

```bash
pip install -r requirements.txt
pytest tests/ -v
python -c "from src.config import cfg; print(cfg.model_dump())"
python -c "from src.data.preprocessing import clean_caption; print(clean_caption(\"Dog's running!!!\"))"
```
