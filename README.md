# Vision2Text

Image captioning system that pairs an **EfficientNetV2-S** encoder with a **Transformer decoder**, served through a **FastAPI** backend and a **React/TypeScript** frontend that lets you hover over each generated word to see exactly where in the image the model was looking.

## Why this repo exists

Vision2Text started as a single Kaggle notebook. Rebuilding it into this repo surfaced three real, silent bugs:

| # | Bug | Fix |
|---|-----|-----|
| 1 | `str.replace("[^a-zA-Z]", "")` on a pandas Series treats the pattern as a **literal string**, not regex — captions were never actually cleaned | Compiled `re.sub` patterns in `src/data/preprocessing.py` |
| 2 | Hardcoded `/kaggle/input/...` paths copy-pasted across 8+ notebook cells | Every path centralized in `src/config.py`, overridable via `.env` |
| 3 | EfficientNet feature extraction re-ran on every notebook restart (3–5 min wasted per session) | Content-addressed `.h5` cache in `src/features/extractor.py`, keyed by an MD5 hash of `(backbone, image_size, data_dir)` |

See [`tasks/lessons.md`](tasks/lessons.md) for the full write-up and [`tasks/plan.md`](tasks/plan.md) for the original scaffolding plan.

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
│  • output projection weight-tied to token embedding│
└───────────────────────────────────────────────────┘
        │
        ▼
BPE vocabulary logits (size 8000)
        │
   greedy decode (training/validation) or beam search (inference)
```

The decoder attends to all 49 spatial patches per generated token rather than a single pooled vector, which is what makes the attention-heatmap visualization in the frontend possible.

## Key features

- **Modular, config-driven design** — every hyperparameter and path lives in a single Pydantic `Config` (`src/config.py`), overridable via `.env` or `V2T_*` environment variables
- **Cached feature extraction** — CNN features are extracted once and stored in an HDF5 file; the cache key changes automatically if you switch backbones or image size
- **BPE tokenizer** — trained from scratch on cleaned captions with HuggingFace `tokenizers`, saved as a portable `vocab.json`
- **PyTorch Lightning training loop** — teacher forcing for training, greedy decoding for validation, BLEU-4 tracked as the checkpoint metric, cosine LR schedule with warmup, gradient clipping, automatic mixed precision
- **Beam search inference** with length normalization, plus decoder cross-attention extraction (via forward hooks that force `need_weights=True`) so every output token can be traced back to the image regions it attended to
- **Production-style FastAPI backend** — in-memory image handling (no disk writes), content-type + size + Pillow-verified image validation, `/health` liveness check, and a Prometheus-compatible `/metrics` endpoint (p50/p95 latency, request count)
- **React + Vite + TypeScript frontend** — upload an image, get a caption, and hover over each word to see its attention heatmap overlaid on the photo
- **Dockerized deployment** — multi-stage build (throwaway build stage, slim non-root runtime, healthcheck) behind an Nginx reverse proxy via `docker-compose`

## Project structure

```
Vision2Text/
├── src/
│   ├── config.py               # Pydantic BaseSettings — single source of truth for paths & hyperparameters
│   ├── data/                   # Dataset loading, caption cleaning, BPE vocab (see note below)
│   │   ├── dataset.py          # FlickrDataset — torch Dataset over Flickr8k/30k
│   │   ├── preprocessing.py    # clean_caption() — lowercase, strip possessives, drop punctuation/numbers
│   │   └── vocab.py            # build_vocab / load_vocab / encode (HuggingFace tokenizers BPE)
│   ├── features/
│   │   └── extractor.py        # Pluggable CNN backbone, content-addressed .h5 feature cache
│   ├── models/
│   │   └── captioner.py        # ImageEncoder, CaptionDecoder, ImageCaptioner (greedy decode included)
│   ├── training/
│   │   └── trainer.py          # PyTorch Lightning module + trainer factory (BLEU-4, checkpointing)
│   ├── inference/
│   │   └── engine.py           # Loads a checkpoint once, runs beam search + attention extraction
│   ├── api/
│   │   ├── app.py              # FastAPI app: /caption, /health, /metrics
│   │   └── schemas.py          # Pydantic response models shared by the API and its tests
│   └── evaluation/             # Reserved for offline metric scripts (BLEU/METEOR/CIDEr sweeps)
├── frontend/                   # React + Vite + TypeScript client
│   └── src/
│       ├── App.tsx
│       ├── api/client.ts       # XHR upload with progress + abort support
│       ├── components/         # ImageUploader, CaptionResult (attention heatmap canvas overlay)
│       └── utils/attention.ts  # Renders the [T, 49] attention grid onto the image
├── tests/                      # pytest suite for vocab + preprocessing
├── tasks/
│   ├── plan.md                 # Original scaffolding plan
│   └── lessons.md              # Bugs found and fixed while porting off the Kaggle notebook
├── train.py                    # Training entry point
├── Dockerfile                  # Multi-stage build for the FastAPI service
├── docker-compose.yml          # api + nginx stack
├── nginx.conf                  # Reverse proxy config
├── requirements.txt            # Training/model dependencies
└── requirements-api.txt        # FastAPI/uvicorn dependencies
```

> **Note:** `src/data/` (`dataset.py`, `preprocessing.py`, `vocab.py`) is the module that `train.py`, `trainer.py`, `inference/engine.py`, and the `tests/` suite all import from — but it isn't present in this copy of the repo. Add it back before running training, inference, or the test suite. `src/evaluation/` is currently an empty package reserved for future metric scripts.

## Getting started

### Prerequisites

- Python 3.11
- Node.js 18+ (frontend only)
- Docker & Docker Compose (optional, for containerized deployment)
- A CUDA-capable GPU is recommended for training but not required for inference

### Backend setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # training/model deps
pip install -r requirements-api.txt    # API deps (only needed to serve the model)
```

### Dataset

Download **Flickr8k** or **Flickr30k** and arrange it so `captions.txt` and the images sit under the `data_dir` configured in `src/config.py` (default: `src/data/`, filename `captions.txt/captions.txt`). Switch datasets or paths via `.env`:

```bash
# .env
V2T_DATASET_NAME=flickr8k
V2T_DATA_DIR=src/data
```

## Configuration

All settings are defined in `src/config.py` and can be overridden with a `V2T_` prefixed environment variable or a `.env` file. Notable defaults:

| Setting | Env var | Default |
|---|---|---|
| Dataset | `V2T_DATASET_NAME` | `flickr8k` |
| Backbone | `V2T_BACKBONE` | `efficientnet_v2_s` |
| Image size | `V2T_IMAGE_SIZE` | `224` |
| Vocab size | `V2T_VOCAB_SIZE` | `8000` |
| Max caption length | `V2T_MAX_CAPTION_LEN` | `40` |
| Model width | `V2T_D_MODEL` | `512` |
| Decoder layers / heads | `V2T_NUM_DECODER_LAYERS` / `V2T_NHEAD` | `4` / `8` |
| Learning rate | `V2T_LEARNING_RATE` | `1e-4` |
| Batch size | `V2T_BATCH_SIZE` | `32` |
| Max epochs | `V2T_MAX_EPOCHS` | `3` |

## Training

```bash
python train.py
```

This builds (or loads) the BPE vocabulary, prepares train/val dataloaders (90/10 split, seeded), then trains with PyTorch Lightning, checkpointing the model with the best `val_bleu4`. On completion it prints the checkpoint path and the environment variables needed to serve it.

## Running the API

```bash
export V2T_CHECKPOINT=outputs/checkpoints/best-XX-0.XXXX.ckpt
export V2T_VOCAB_PATH=cache/vocab.json
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Caption an image:

```bash
curl -X POST http://localhost:8000/caption \
     -F "file=@/path/to/image.jpg" \
     -F "beam_size=4"
```

Response shape:

```json
{
  "caption": "a dog running through a field",
  "beam_size": 4,
  "inference_ms": 182.4,
  "tokens": ["a", "dog", "running", "through", "a", "field"],
  "attention_weights": [[...49 floats...], ...]
}
```

Other endpoints: `GET /health` (readiness) and `GET /metrics` (Prometheus text format — request count and p50/p95 latency).

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Upload a photo and the UI displays the generated caption with each token hoverable — hovering renders that token's 7×7 attention grid as a heatmap over the source image, so you can see which region the model relied on for that word.

## Docker deployment

```bash
mkdir -p checkpoints cache
cp /path/to/best.ckpt checkpoints/best.ckpt
cp /path/to/vocab.json cache/vocab.json
docker compose up --build
```

This starts the FastAPI service (non-root user, healthcheck-gated) behind an Nginx reverse proxy on port 80. Model weights are mounted read-only rather than baked into the image, so you can swap checkpoints without rebuilding.

## Testing

```bash
pytest tests/ -v
```

Covers BPE vocabulary building/round-tripping/encoding (`test_vocab.py`) and caption cleaning — lowercasing, possessive stripping, punctuation/number removal, whitespace collapsing (`test_preprocessing.py`).

## Tech stack

**Model & training:** PyTorch, torchvision (EfficientNetV2-S), PyTorch Lightning, HuggingFace `tokenizers` & `transformers`, torchmetrics
**Serving:** FastAPI, Uvicorn, Pydantic
**Frontend:** React 18, TypeScript, Vite
**Infra:** Docker, Nginx, Prometheus-compatible metrics

## Acknowledgments

- [Flickr8k / Flickr30k](https://www.kaggle.com/datasets) datasets
- [EfficientNetV2](https://arxiv.org/abs/2104.00298) (Tan & Le)
- HuggingFace `tokenizers` and `transformers`
- PyTorch Lightning
