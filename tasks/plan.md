# Vision2Text — Plan & Status

## Context

Started from a monolithic Kaggle notebook with silent bugs (regex cleaning that did nothing,
hardcoded paths, repeated feature extraction, no structure). The repo splits it into a tested,
config-driven pipeline plus a serving stack. The README is the user-facing description; this file
records the module responsibilities and what remains.

---

## Directory layout (current)

```
src/
  config.py            Pydantic BaseSettings — every path & hyperparameter
  data/
    preprocessing.py   clean_caption / clean_captions  (compiled re patterns)
    vocab.py           BPE (HF tokenizers): build_vocab, load_vocab, encode, special_ids, group_pieces
    dataset.py         parse_captions, split_image_names, load_splits, FlickrDataset (pairs),
                       FlickrEvalDataset (image + all references), get_dataloader
    transforms.py      build_transform — shared by training and serving
    download.py        Flickr8k via kagglehub
    toy.py             synthetic dataset for tests / smoke runs
  models/captioner.py  ImageEncoder (probed patch grid), CaptionDecoder (pre-norm + final LN, tied embeddings),
                       ImageCaptioner, greedy_decode
  training/
    trainer.py         CaptioningModule (teacher forcing; greedy-decode BLEU-4 vs all references),
                       build_trainer (checkpoint, early stopping, CSV logs)
    export.py          Lightning .ckpt -> self-contained model.pt
  inference/engine.py  InferenceEngine: beam search, thread-safe attention capture, word grouping
  evaluation/          metrics.py (BLEU, ROUGE-L, CIDEr-D, METEOR), evaluate.py (CLI)
  api/                 FastAPI app + schemas
frontend/              React + Vite + TS (Vitest, ESLint)
tests/                 data, model, engine, API, metrics, pipeline (train -> export -> serve -> evaluate -> resume)
train.py               training entry point (--resume, --rebuild-vocab)
```

---

## Status

Done and verified (tests / runs on the authoring machine):
- data pipeline on real Flickr8k (8,091 images, 5 captions each, 6473/809/809 split, no overlap)
- training loop, validation BLEU, checkpointing, export, resume (synthetic data + 200 real images)
- API served live from an exported model, real image through HTTP, error paths, metrics
- 111 backend tests, ruff, mypy; 30 frontend tests, typecheck, lint, production build, `npm ci`

Written but **not executed** anywhere yet:
- Dockerfiles / docker-compose / nginx.conf (no Docker on the authoring machine) — CI builds the images
- GitHub Actions workflow
- Flickr30k parsing against the real dataset (tested on synthetic files only)
- visual check of the UI changes in a browser (logic is unit-tested, pixels were not eyeballed)

## Remaining (needs the owner)

1. **Full training run on a GPU** and recording the metrics in the README "Results" section.
   CPU is ~55 min/epoch, so a real run on CPU is impractical.
2. **Decide where trained models live** (GitHub Release / Hugging Face Hub / DVC). `model.pt` is a single file,
   ~170 MB for the default size.
3. **Choose a LICENSE.**
4. Run the Docker stack once (`docker compose up --build`) and confirm CI is green.

## Possible later improvements
- KV-cache for beam search; higher input resolution (EfficientNetV2-S was tuned for 384); EMA of weights
- frozen-encoder mode with a cached spatial-feature file for fast ablations
- official METEOR 1.5 / SPICE via `pycocoevalcap` if published-table comparability matters
- dependency lock file (`uv`/`pip-tools`)
