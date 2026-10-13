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

**Fix:** Use `re.sub` with compiled patterns (`src/data/preprocessing.py`).

**Why it matters:** This is a completely silent failure — no error, no warning, wrong output.
All downstream vocab and model training was built on dirty captions.

---

## Bug 2 — Hardcoded Kaggle dataset paths

**What happened:** Paths like `/kaggle/input/flickr8k/Images/` were copy-pasted across 8+ cells.

**Fix:** Every path lives in `src/config.py`; override via `.env` or environment variable.
Nothing in `src/` or `tests/` should contain a hardcoded path.

---

## Bug 3 — Feature extraction ran on every notebook restart (and why the cache was removed)

**What happened:** The notebook ran the EfficientNet forward pass over all 8,000 images on
every kernel restart (3–5 minutes).

**First fix (later removed):** a content-addressed `.h5` feature cache. It turned out to be
dead code: nothing called it, and it stored the *pooled* 1280-d vector while the decoder needs
the un-pooled 7×7×1280 grid. More fundamentally, this project fine-tunes the CNN end-to-end,
so there is no frozen feature set to cache. The module was deleted (it is in git history).
If a frozen-encoder mode is ever wanted, cache the `.features` output (49×1280, fp16) instead.

---

## Bug 4 — `src/data/` was never committed

**What happened:** `.gitignore` contained `data/`. An unanchored pattern matches a directory
of that name at *any depth*, so it also swallowed the source package `src/data/`
(dataset, preprocessing, vocab). No commit ever contained them; the repo could not run or test.

**Fix:** anchor it (`/data/`), move the dataset out of the package (`data/`), recreate the code.

**Lesson:** when ignoring data directories, anchor the pattern, and run
`git check-ignore -v <file>` on source files you expect to be tracked.

---

## Bug 5 — Decoder started at loss ≈ 76 instead of ≈ ln(vocab)

**What happened:** A smoke run showed `train_loss` of 76 → 107 in the first epochs for a
300-token vocabulary (should start near ln 300 ≈ 5.7). Two causes combined:
the decoder used pre-norm layers but **no final LayerNorm**, and the output projection is
**tied to an `nn.Embedding` initialised N(0, 1)**, so initial logits had a standard deviation
around √d_model. With `d_model=512` it would have been worse.

**Fix:** final `LayerNorm` on the decoder stack + embedding init `N(0, 0.02)`.
After: first-step loss 4.6, epoch losses 5.2 → 2.5 and rising BLEU on the toy set.
Regression test: `test_initial_loss_is_near_chance`.

**Lesson:** always look at the loss of step 0. Static review and unit tests of shapes cannot see this.

---

## Bug 6 — Lightning checkpoints could not be loaded safely

`save_hyperparameters(cfg.model_dump())` pickled `pathlib.Path` objects into the checkpoint, which
`torch.load(weights_only=True)` refuses (and a checkpoint trained on Windows holds `WindowsPath`,
unloadable on Linux at all). Fixes: `model_dump(mode="json")`; the trainer embeds the
architecture + tokenizer in every checkpoint; `train.py` exports a plain-data `model.pt`.

## Bug 7 — Serving read its architecture from the environment, not the checkpoint

The engine rebuilt the network from whatever `V2T_*` variables the server happened to have, so
a mismatch loaded "fine" (or failed obscurely). Now the architecture and tokenizer travel inside
the checkpoint and the environment cannot override them.

## Bug 8 — Race condition in attention extraction

`caption()` registered/removed forward hooks on the *shared* model per request while the API ran
two inference threads, so one request's attention pass could capture another request's tensors.
Hooks are now installed once and only act for a thread that opted in through a `threading.local`.
Regression test: `test_concurrent_requests_do_not_corrupt_each_other`.

## Bug 9 — Smaller ones

- Serving downloaded ImageNet weights at startup just to overwrite them (breaks in locked-down containers) → `pretrained=False`.
- Beam search: a hypothesis that emitted EOS first had length 0 → `0 ** 0.7` division by zero.
- Validation BLEU used one reference per hypothesis (Flickr has five) → evaluation is per image against all references.
- PAD id was hard-coded 0 in the model but looked up elsewhere → `special_ids()` + PAD guaranteed 0 + a test.
- `.env` containing the serving variables made `Config()` raise (extra keys forbidden) → they are real fields now, extras ignored.
- Decoder mixed a float causal mask with a bool padding mask (deprecated) → both float.

---

## Decision — HuggingFace `tokenizers` over `nltk` word-level vocab

- Handles out-of-vocabulary words gracefully (subword units)
- `vocab.json` is a self-contained, portable artifact — and is additionally embedded in `model.pt`
- Words are marked with an end-of-word suffix (`</w>`) so pieces can be merged back into words for display

## Decision — Pydantic v2 `BaseSettings` over `argparse` / `yaml`

- Type validation is free; `.env` override means no code changes per environment
- `cfg.model_dump()` gives a serialisable snapshot of every run's config

## Decision — implement CIDEr-D / ROUGE-L instead of depending on `pycocoevalcap`

`pycocoevalcap` needs Java and ships large jars; the two metrics are ~80 lines. Trade-off:
numbers are not guaranteed bit-identical to the official toolkit, so the README says so.

## Decision — no horizontal-flip augmentation

Captions mention "left"/"right"; a flip would silently make some of them wrong.

## Decision — no API key on the web app

The browser would have to ship the key, so it protects nothing. Abuse is limited at the edge
instead (Nginx `limit_req`), and `/metrics` is not exposed publicly.
