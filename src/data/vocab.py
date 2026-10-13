"""
BPE vocabulary built with HuggingFace ``tokenizers``.

Design notes
------------
* The special tokens are registered first, so **PAD is always id 0**. The
  decoder's ``padding_idx`` and its padding mask rely on that.
* Words are marked with an end-of-word suffix (``</w>``). That makes sub-word
  pieces unambiguous (``runn`` + ``ing</w>``) so the inference engine can fold
  pieces back into whole words for display, and lets the ``BPEDecoder`` turn
  ids back into normal spaced text.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
WORD_END = "</w>"


def build_vocab(captions: list[str], vocab_size: int, save_path: Path) -> Tokenizer:
    """Train a BPE tokenizer on already-cleaned captions and save it as JSON."""
    tokenizer = Tokenizer(
        models.BPE(unk_token=UNK_TOKEN, end_of_word_suffix=WORD_END)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.BPEDecoder(suffix=WORD_END)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,     # order matters: PAD -> id 0
        end_of_word_suffix=WORD_END,
        show_progress=False,
    )
    tokenizer.train_from_iterator(captions, trainer)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def load_vocab(path: Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {path}. Run `python train.py` to build it."
        )
    return Tokenizer.from_file(str(path))


def special_ids(tokenizer: Tokenizer) -> tuple[int, int, int]:
    """``(pad_id, bos_id, eos_id)``, failing loudly if the vocab lacks one.

    ``Tokenizer.token_to_id`` returns ``None`` for an unknown token; letting that
    flow into the model would surface much later as a confusing tensor error.
    """
    ids = []
    for token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(
                f"Vocabulary has no {token!r} token; rebuild it with "
                "`python train.py --rebuild-vocab`."
            )
        ids.append(token_id)
    return ids[0], ids[1], ids[2]


def encode(tokenizer: Tokenizer, text: str, max_len: int) -> list[int]:
    """``[BOS] tokens [EOS] [PAD]...`` of length exactly ``max_len``.

    Over-long captions are truncated *before* the EOS is appended, so BOS is
    always first and EOS is always present.
    """
    if max_len < 2:
        raise ValueError("max_len must leave room for BOS and EOS (>= 2)")

    pad_id, bos_id, eos_id = special_ids(tokenizer)
    ids = tokenizer.encode(text).ids[: max_len - 2]
    out = [bos_id, *ids, eos_id]
    out += [pad_id] * (max_len - len(out))
    return out


def group_pieces(pieces: list[str]) -> list[tuple[str, list[int]]]:
    """Fold BPE pieces into whole words.

    ``["a</w>", "runn", "ing</w>"]`` -> ``[("a", [0]), ("running", [1, 2])]``.
    Returns each word together with the indices of the pieces that form it, so
    per-piece attention rows can be merged into one row per word.
    """
    words: list[tuple[str, list[int]]] = []
    text, idxs = "", []
    for i, piece in enumerate(pieces):
        text += "?" if piece == UNK_TOKEN else piece.removesuffix(WORD_END)
        idxs.append(i)
        if piece.endswith(WORD_END):
            words.append((text, idxs))
            text, idxs = "", []
    if idxs:   # sequence cut off mid-word (e.g. hit max length)
        words.append((text, idxs))
    return words
