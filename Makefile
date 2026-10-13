# Convenience wrappers. On Windows without `make`, run the commands directly
# (they are all plain `python ...` / `npm ...` invocations).

PY ?= python

.PHONY: install data train evaluate serve test lint check frontend-install frontend-dev frontend-check docker-up

install:            ## training + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:               ## download Flickr8k into ./data (via kagglehub)
	$(PY) -m src.data.download

train:              ## train, then export outputs/model.pt
	$(PY) train.py

evaluate:           ## BLEU/ROUGE-L/CIDEr-D on the held-out test split
	$(PY) -m src.evaluation.evaluate --checkpoint outputs/model.pt

serve:              ## run the API on :8000 with outputs/model.pt
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

check: lint          ## everything CI runs for the backend
	$(PY) -m mypy
	$(PY) -m pytest -q

frontend-install:
	cd frontend && npm ci

frontend-dev:       ## Vite dev server on :5173 (proxies to the API on :8000)
	cd frontend && npm run dev

frontend-check:     ## everything CI runs for the frontend
	cd frontend && npm run typecheck && npm run lint && npm test && npm run build

docker-up:          ## needs checkpoints/model.pt
	docker compose up --build
