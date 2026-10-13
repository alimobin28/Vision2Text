# Vision2Text — Work Status

_Updated 2026-10-08. This file started as the list of everything that was left; it now records what was done, what was skipped on purpose, and what is still open._

**How things were verified.** The work was done in a throwaway Python 3.12 + Node 22 environment (outside the repo). Unless marked otherwise, an item is backed by a passing test or a real run:

- Backend: `ruff` clean, `mypy` clean, **111 tests pass** (~45 s on CPU).
- Frontend: typecheck + lint clean, **30 tests pass**, production build OK, `npm ci` OK.
- Real runs: Flickr8k downloaded and parsed; full training pipeline on a synthetic dataset and on 200 real images with the default model; API served live and called over HTTP with a real photo.

---

## Still open (needs you)

- [ ] **Train on a GPU for real and record the metrics.** Measured here: ~3.2 s per batch of 32 on CPU → ~55 min per Flickr8k epoch, so a real run on CPU takes over a day. The README "Results" section is deliberately empty until there are real numbers. See README → *How the model is trained*.
- [ ] **Decide where trained models live** (GitHub Release / Hugging Face Hub / DVC). `outputs/model.pt` is one self-contained ~170 MB file; `.gitignore` excludes it.
- [ ] **Choose a LICENSE** (I did not pick one for you).
- [ ] **Run the Docker stack once** (`docker compose up --build`) and confirm the GitHub Actions run is green. The Dockerfiles, compose files, `nginx.conf` and the workflow were written and reviewed but **never executed** — Docker is not available on the authoring machine.
- [ ] **Look at the UI in a browser.** The new beam-size selector, status banner and pin-a-word behaviour are unit-tested, but nobody has eyeballed the rendering.
- [ ] Flickr30k: the parser is implemented but only tested on synthetic files.

---

## Done

### P0 — blockers
- [x] **B1** `src/data/` recreated (`preprocessing`, `vocab`, `dataset`, `transforms`) to the contract of the existing callers and tests; `.gitignore` rule anchored to `/data/` (it was the cause); dataset moved out of the package (`data/`). The 25 pre-existing tests, never runnable before, pass.
- [x] **B2** end-to-end run demonstrated: train → checkpoint → export → serve → evaluate → resume (tests + real runs). *Full* training is open above.
- [x] Dataset download via `kagglehub` (`python -m src.data.download`, slug `adityajn105/flickr8k`) — verified: 8,091 images, 40,455 captions.

### P1 — bugs
- [x] **C1** checkpoint unloadable under `weights_only=True` → hyperparameters saved as JSON; checkpoints embed architecture + tokenizer; `train.py` exports a plain-data `model.pt`. Test loads a real Lightning `.ckpt` with `weights_only=True`.
- [x] **C2** no ImageNet download at serve time (`pretrained=False`); test spies on `get_model`.
- [x] **C3** attention-hook race → hooks installed once, thread-local capture; concurrency test.
- [x] **C4** feature cache: **deleted** (dead code, wrong shape, CNN is fine-tuned end-to-end). In git history if you want it back. `h5py`/`datasets` dropped from requirements.
- [x] **C5** PAD id: guaranteed 0, passed into the model, `special_ids()` fails loudly if a token is missing; vocab-size check.
- [x] **C6** validation BLEU now per image against all 5 references.
- [x] **C7** serving settings are real `Config` fields; unknown `.env` keys ignored; `.env.example` restored.
- [x] **C8** one public split function used by vocab building and dataloaders.
- [x] **C9** `max_epochs` back to 30 (+ early stopping); best checkpoint from the callback (not filename sort); `seed_everything`; unused imports; `except` clean-up; no hard-coded paths.
- [x] **New, found by running it:** decoder had no final LayerNorm and N(0,1) tied embeddings → initial loss 76 instead of ≈ ln(vocab) (now 4.6 on the toy run). Regression test added.
- [x] **New:** beam search divide-by-zero on an empty hypothesis; decoder mask-type deprecation.

### P2 — features
- [x] **D1** slim CPU API image from `requirements-infer.txt` (+ `requirements.txt`, `requirements-dev.txt`), `.dockerignore`, GPU override file. _Not built here._
- [x] **D2** frontend served in production: `frontend/Dockerfile` (Node build → Nginx), SPA routing, asset caching. _Not built here._
- [x] **D3** `/metrics` returns 404 through Nginx (scrape `api:8000` internally).
- [x] **D4** rate limiting (Nginx), pixel-count/decompression-bomb guard, EXIF rotation, failures counted in metrics, request logging, 503 when the model isn't loaded.
- [x] **D5** `.python-version`, `requires-python`.
- [x] **E1** `src/evaluation`: BLEU-1..4, ROUGE-L, CIDEr-D (own implementations), optional NLTK METEOR; CLI writes `metrics_<split>.json` and `predictions_<split>.json`.
- [x] **E2** train/val/test split by image (80/10/10).
- [x] **E3** Flickr30k parser (synthetic-tested only).
- [x] **E4** readable tokens: BPE pieces folded into whole words; attention rows averaged per word.
- [x] **F1** frontend builds. **F2** beam-size selector. **F3** backend status banner with retry. **F4** heatmap aligned to patch centres; grid size derived from the data. **F5** Vitest + Testing Library + ESLint, 30 tests. **F6** hover previews, click/tap pins, Escape clears.
- [x] **G1** tests for model, engine, API, metrics, dataset, config, full pipeline.
- [x] **G2** (partly) lower LR for the pretrained CNN, mild augmentation (no flips, on purpose), early stopping, seeding, CSV logs.
- [x] **G3** GitHub Actions workflow (backend, frontend, Docker builds). _Never run._
- [x] **G4** `pyproject.toml`, `.env.example`, `Makefile`, `.python-version`.
- [x] **G6** dataset download helper.
- [x] **Extra:** `train.py --resume` (+ `last.ckpt` each epoch) so a long run can be continued after an interruption.

### Documentation
- [x] README rewritten to match the code (training explanation, evaluation, Docker, testing); `tasks/lessons.md` and `tasks/plan.md` updated.

---

## Skipped on purpose

| Item | Why |
|---|---|
| API key / auth for the web app | A browser app would have to ship the key, so it protects nothing; abuse is limited with Nginx rate limiting instead. |
| Prettier | Would reformat every existing frontend file for no functional gain. |
| EMA, 384-px input, W&B/TensorBoard | Experiments to try *after* a baseline exists; not needed for a working system. |
| KV-cache beam search (G5) | Fine at 40 tokens; revisit only if CPU latency under load matters. |
| Dependency lock file | Needs a decision between `uv` / `pip-tools`; versions are lower-bounded for now. |
| Official METEOR 1.5 / SPICE | Need Java; the repo uses its own BLEU/ROUGE-L/CIDEr-D and says so. |
