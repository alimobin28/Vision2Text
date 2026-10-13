# Vision2Text

Image captioning system that pairs an **EfficientNetV2-S** encoder with a **Transformer decoder**, served through a **FastAPI** backend and a **React/TypeScript** frontend that lets you hover over each generated word to see exactly where in the image the model was looking.

Everything runs locally — no third-party or paid API is involved. The only downloads are the dataset (Kaggle) and the pretrained EfficientNet weights (PyTorch), both fetched once.

## Why this repo exists

Vision2Text started as a single Kaggle notebook. Rebuilding it into this repo surfaced silent bugs:

| # | Bug | Fix |
|---|-----|-----|
| 1 | `str.replace("[^a-zA-Z]", "")` on a pandas Series treats the pattern as a **literal string**, not regex — captions were never actually cleaned | Compiled `re.sub` patterns in `src/data/preprocessing.py` |
| 2 | Hardcoded `/kaggle/input/...` paths copy-pasted across 8+ notebook cells | Every path centralized in `src/config.py`, overridable via `.env` |
| 3 | EfficientNet feature extraction re-ran on every notebook restart | The notebook froze the CNN; this project fine-tunes it end-to-end instead, so there is nothing to cache (see `tasks/lessons.md`) |

Further bugs found while making the pipeline actually run end-to-end (a decoder whose initial loss was ~76 instead of ~ln(vocab), an unloadable checkpoint format, a thread race in the attention extractor, …) are written up in [`tasks/lessons.md`](tasks/lessons.md).

## Architecture

```
Image (224×224×3)
        │
        ▼
EfficientNetV2-S conv trunk (pretrained, torchvision)
        │   (B, 1280, 7, 7)
        ▼
Flatten → Linear projection → 49 patch embeddings @ d_model=512
        │   + learnable positional embeddings
        ▼
┌───────────────────────────────────────────────────┐
│ 4-layer Transformer decoder (8 heads, pre-norm)    │
│  • causal self-attention over generated tokens     │
│  • cross-attention over the 49 image patches       │
│  • final LayerNorm                                 │
│  • output projection weight-tied to token embedding│
└───────────────────────────────────────────────────┘
        │
        ▼
BPE vocabulary logits (size 8000)
        │
   greedy decode (validation) or beam search (inference / evaluation)
```

The decoder attends to all spatial patches per generated token rather than a single pooled vector, which is what makes the attention-heatmap visualization possible. The patch grid is *probed* from the backbone (not hard-coded), so other `torchvision` backbones with a `.features` trunk work too.

## Key features

- **Modular, config-driven design** — every hyperparameter and path lives in one Pydantic `Config` (`src/config.py`), overridable via `.env` or `V2T_*` environment variables
- **Leak-free data pipeline** — train/val/test split *by image* (never by caption), one seeded function used by both the vocabulary builder and the dataloaders
- **BPE tokenizer** trained from scratch on training captions only; sub-word pieces are folded back into whole words for the UI
- **PyTorch Lightning training** — teacher forcing, label smoothing, AdamW with a lower LR for the pretrained CNN, cosine schedule with warm-up, gradient clipping, automatic mixed precision, early stopping, **resumable** (`--resume`)
- **Honest validation** — greedy-decoded captions scored with BLEU-4 against *all five* references of each image; best checkpoint is chosen on that
- **Self-contained model file** — `outputs/model.pt` holds weights + architecture + tokenizer, loads with `torch.load(weights_only=True)`, and can't disagree with the serving environment's settings
- **Offline evaluation** — BLEU-1..4, ROUGE-L, CIDEr-D (optionally METEOR) on the held-out test split
- **Beam search inference** with length normalization, plus decoder cross-attention extraction (via forward hooks, thread-safe) so every output word can be traced back to image regions
- **Production-style FastAPI backend** — in-memory image handling, content-type + size + pixel-count + Pillow validation, EXIF-orientation handling, `/health` readiness (503 until the model is loaded), Prometheus-style `/metrics`
- **React + Vite + TypeScript frontend** — upload, choose beam size, get a caption; hover/focus a word to preview its heatmap, click/tap to pin it
- **Dockerized deployment** — CPU-only API image, Nginx serving the built UI, rate limiting, `/metrics` not exposed publicly

## Project structure

```
Vision2Text/
├── src/
│   ├── config.py               # Pydantic BaseSettings — single source of truth
│   ├── data/
│   │   ├── preprocessing.py    # clean_caption() — lowercase, punctuation/numbers out
│   │   ├── vocab.py            # BPE tokenizer: build / load / encode / group pieces into words
│   │   ├── dataset.py          # caption parsing, image-level splits, train & eval datasets, dataloaders
│   │   ├── transforms.py       # image transforms shared by training and serving
│   │   ├── download.py         # python -m src.data.download  (Flickr8k via kagglehub)
│   │   └── toy.py              # tiny synthetic dataset for tests / smoke runs
│   ├── models/captioner.py     # ImageEncoder, CaptionDecoder, ImageCaptioner (+ greedy decode)
│   ├── training/
│   │   ├── trainer.py          # Lightning module + trainer factory
│   │   └── export.py           # .ckpt -> self-contained model.pt
│   ├── inference/engine.py     # loads model.pt once; beam search + attention extraction
│   ├── evaluation/
│   │   ├── metrics.py          # BLEU, ROUGE-L, CIDEr-D, (METEOR)
│   │   └── evaluate.py         # python -m src.evaluation.evaluate
│   └── api/                    # FastAPI app + Pydantic response schemas
├── frontend/                   # React + Vite + TypeScript client (+ Vitest, ESLint)
├── tests/                      # pytest: data, model, engine, API, metrics, end-to-end pipeline
├── train.py                    # training entry point
├── Dockerfile                  # API image (CPU torch by default)
├── frontend/Dockerfile         # builds the UI and serves it with Nginx
├── docker-compose.yml          # api + web;  docker-compose.gpu.yml = GPU override
├── nginx.conf                  # static files, /caption proxy + rate limit, /metrics hidden
├── requirements-infer.txt      # serving only (what the Docker image installs)
├── requirements.txt            # + training & evaluation
├── requirements-dev.txt        # + pytest, ruff, mypy
├── pyproject.toml              # pytest / ruff / mypy config
└── tasks/                      # plan.md, lessons.md
```

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- Node.js 20.19+ / 22 (frontend only)
- Docker & Docker Compose (optional)
- A CUDA GPU is strongly recommended for training (see [Training time](#training-time)); inference runs fine on CPU

### Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # training + evaluation + tests
```

For a CPU-only machine, install the small PyTorch wheels first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

### Dataset

```bash
python -m src.data.download          # Flickr8k (adityajn105/flickr8k) via kagglehub -> ./data
```

That is the same as:

```python
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
```

followed by copying the files into `data/`. The expected layout is

```
data/
├── Images/            # 8,091 .jpg files
└── captions.txt       # "image,caption" CSV, 5 captions per image
```

If the files live elsewhere, set `V2T_DATA_DIR` (and `V2T_CAPTIONS_FILE` / `V2T_IMAGES_SUBDIR` if the names differ). Flickr30k is supported via `V2T_DATASET_NAME=flickr30k` (`results.csv` + `flickr30k_images/`), but that parser has only been tested on synthetic data.

## Configuration

All settings are defined in `src/config.py` and can be overridden with a `V2T_`-prefixed environment variable or a `.env` file (see [`.env.example`](.env.example)). Notable defaults:

| Setting | Env var | Default |
|---|---|---|
| Dataset / directory | `V2T_DATASET_NAME` / `V2T_DATA_DIR` | `flickr8k` / `data` |
| Split (val / test fraction) | `V2T_VAL_FRACTION` / `V2T_TEST_FRACTION` | `0.1` / `0.1` |
| Backbone | `V2T_BACKBONE` | `efficientnet_v2_s` |
| Image size | `V2T_IMAGE_SIZE` | `224` |
| Vocab size / max caption length | `V2T_VOCAB_SIZE` / `V2T_MAX_CAPTION_LEN` | `8000` / `40` |
| Model width / layers / heads | `V2T_D_MODEL` / `V2T_NUM_DECODER_LAYERS` / `V2T_NHEAD` | `512` / `4` / `8` |
| Learning rate / CNN multiplier | `V2T_LEARNING_RATE` / `V2T_BACKBONE_LR_MULT` | `1e-4` / `0.1` |
| Batch size | `V2T_BATCH_SIZE` | `32` |
| Max epochs / early-stop patience | `V2T_MAX_EPOCHS` / `V2T_EARLY_STOPPING_PATIENCE` | `30` / `5` |
| Quick runs: use only N images | `V2T_MAX_IMAGES` | `0` (all) |
| Serving: model file / device / threads | `V2T_CHECKPOINT` / `V2T_DEVICE` / `V2T_WORKERS` | `outputs/model.pt` / `auto` / `2` |

## How the model is trained

1. **Split.** All image names are sorted, shuffled with a fixed seed and cut into train / val / test (80 / 10 / 10) **by image**, so the five captions of a photo are never spread across splits. On Flickr8k that is 6,473 / 809 / 809 images (32,365 training caption pairs).
2. **Clean + tokenize.** Captions are lowercased, punctuation and bare numbers removed. A byte-pair-encoding vocabulary (8,000 tokens) is trained **on the training captions only**. Each caption becomes `[BOS] tokens [EOS] [PAD]…` of fixed length 40.
3. **Forward pass.** EfficientNetV2-S (ImageNet-pretrained) turns the 224×224 image into a 7×7 grid of 1280-d features → 49 patch vectors of size 512. The Transformer decoder predicts the caption token by token, attending to all 49 patches at every step.
4. **Loss — teacher forcing.** During training the decoder is fed the *true* previous tokens and must predict the next one at every position; the loss is cross-entropy with label smoothing 0.1, ignoring padding.
5. **Optimisation.** AdamW (weight decay 0.01, none on biases/norms/embeddings), learning rate 1e-4 for the new layers and 1e-5 for the pretrained CNN (`BACKBONE_LR_MULT`), linear warm-up then cosine decay, gradient clipping at 1.0, bf16/fp16 mixed precision on GPU. Training images get a mild random crop + colour jitter (no horizontal flip: captions say "left"/"right").
6. **Validation — greedy decoding.** After every epoch the model *generates* a caption for each validation image and BLEU-4 is computed against all five human captions. The checkpoint with the best BLEU-4 is kept; training stops early after 5 epochs without improvement.
7. **Export.** The best checkpoint is converted into `outputs/model.pt`.

```bash
python train.py                  # trains; writes outputs/checkpoints/*.ckpt, outputs/logs/, outputs/model.pt
python train.py --resume         # continue an interrupted run from outputs/checkpoints/last.ckpt
python train.py --rebuild-vocab  # retrain the BPE vocabulary (do this if you change the dataset)
```

Metrics per step/epoch are written to `outputs/logs/version_N/metrics.csv`. Keep `V2T_MAX_EPOCHS` unchanged when resuming (the LR schedule depends on it).

**Smoke test first** (about two minutes on CPU, needs no download):

```bash
python -m src.data.toy toy_data
V2T_DATA_DIR=toy_data V2T_IMAGE_SIZE=128 V2T_VOCAB_SIZE=300 V2T_MAX_CAPTION_LEN=16 \
V2T_BATCH_SIZE=8 V2T_NUM_WORKERS=0 V2T_MAX_EPOCHS=6 V2T_D_MODEL=128 V2T_NHEAD=4 \
V2T_NUM_DECODER_LAYERS=2 V2T_DIM_FEEDFORWARD=256 V2T_LEARNING_RATE=3e-4 python train.py
```

(on Windows PowerShell set the variables with `$env:V2T_DATA_DIR="toy_data"; …` first).

### Training time

Measured on the development CPU: **≈3.2 s per batch of 32** at 224 px with the default model. One Flickr8k epoch is 1,011 steps ≈ **55 minutes, plus a few minutes of validation** — a full 30-epoch run on CPU would take over a day. Use a CUDA GPU (or a free Colab/Kaggle GPU) for real training; `python train.py` picks it up automatically. A GPU epoch typically takes minutes, but that figure is an estimate, not a measurement from this repo.

## Evaluation

```bash
python -m src.evaluation.evaluate --checkpoint outputs/model.pt          # test split, beam 4
python -m src.evaluation.evaluate --checkpoint outputs/model.pt --split val --meteor
```

Writes `outputs/metrics_<split>.json` and `outputs/predictions_<split>.json` (hypothesis + references per image). **Use the same data settings (`V2T_SEED`, fractions, `V2T_MAX_IMAGES`, `V2T_DATA_DIR`) as in training** — the split is a pure function of them, and different values could let "test" images leak in from training.

CIDEr-D and ROUGE-L are our own implementations of the coco-caption definitions (no Java needed); METEOR uses NLTK's, which is not the official METEOR 1.5. Treat numbers as "this repo's metric implementation", not as drop-in comparable to published tables.

### Results

_Not recorded yet._ Run a full training and the evaluation above, then paste `outputs/metrics_test.json` here (with the git commit and `V2T_*` settings used).

## Running the API

```bash
V2T_CHECKPOINT=outputs/model.pt uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

`model.pt` already contains the tokenizer and architecture; no separate vocab file or matching `V2T_D_MODEL` etc. is needed. Startup takes a couple of seconds and downloads nothing.

```bash
curl -X POST http://localhost:8000/caption -F "file=@/path/to/image.jpg" -F "beam_size=4"
```

```json
{
  "caption": "a dog running through a field",
  "beam_size": 4,
  "inference_ms": 182.4,
  "tokens": ["a", "dog", "running", "through", "a", "field"],
  "attention_weights": [[...49 floats...], ...]
}
```

`tokens` are whole words; `attention_weights[i]` is the 7×7 map (flattened) for `tokens[i]`. Other endpoints: `GET /health` (200 when the model is loaded, 503 while loading) and `GET /metrics` (Prometheus text: request count, error count, p50/p95 latency).

Uploads are rejected with 422 when they are not JPEG/PNG/WEBP, larger than 10 MB, larger than 40 megapixels, or not decodable.

## Frontend

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173, proxies /caption and /health to the API on :8000
```

Upload a photo, pick a beam size and the UI shows the caption with every word as a button: hovering or focusing a word previews its attention heatmap over the photo; clicking/tapping pins it (so it works on touch screens); Escape clears. A banner appears if the API is unreachable or the model is still loading.

Other scripts: `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`.

## Docker deployment

```bash
mkdir -p checkpoints && cp outputs/model.pt checkpoints/model.pt
docker compose up --build            # http://localhost
# GPU host: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Two containers: `api` (non-root, CPU PyTorch, healthcheck-gated, not published) and `web` (Nginx serving the built UI on port 80 and proxying `/caption` and `/health`). The model is mounted read-only, so swapping it needs no rebuild. Nginx rate-limits `/caption` to 30 requests/minute per IP and returns 404 for `/metrics`; scrape metrics from inside the Docker network at `http://api:8000/metrics`.

> The Dockerfiles and compose files were written and reviewed but **not built in the environment they were authored in** (no Docker available). CI builds both images on every push.

## Testing

```bash
pytest -q                 # 111 tests, under a minute on CPU, no dataset or network needed
ruff check . && mypy      # lint + types
cd frontend && npm run typecheck && npm run lint && npm test && npm run build
```

Backend tests cover caption cleaning, vocabulary, dataset parsing/splitting (including no-leakage), model shapes and causality, the inference engine (including concurrency and empty captions), the API (validation, errors, metrics), the metrics, and a full train → export → serve → evaluate → resume pipeline on a synthetic dataset. CI (`.github/workflows/ci.yml`) runs all of the above plus the Docker builds.

## Tech stack

**Model & training:** PyTorch, torchvision (EfficientNetV2-S), PyTorch Lightning, HuggingFace `tokenizers`, torchmetrics
**Serving:** FastAPI, Uvicorn, Pydantic
**Frontend:** React 18, TypeScript, Vite, Vitest, ESLint
**Infra:** Docker, Nginx, GitHub Actions, Prometheus-style metrics

## Acknowledgments

- [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset
- [EfficientNetV2](https://arxiv.org/abs/2104.00298) (Tan & Le)
- HuggingFace `tokenizers`, PyTorch Lightning
