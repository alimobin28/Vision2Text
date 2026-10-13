import json
from pathlib import Path

import pytest

from src.data.vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    WORD_END,
    build_vocab,
    encode,
    group_pieces,
    load_vocab,
    special_ids,
)

_CORPUS = [
    "a dog runs in the park",
    "a cat sits on a mat",
    "two birds fly over the lake",
    "the child laughs and plays",
    "a man walks down the street",
]


@pytest.fixture(scope="module")
def vocab_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocab.json"
    build_vocab(_CORPUS, vocab_size=200, save_path=path)
    return path


@pytest.fixture(scope="module")
def tokenizer(vocab_file):
    return load_vocab(vocab_file)


def test_vocab_file_created(vocab_file):
    assert vocab_file.exists()
    # sanity: it's valid JSON
    data = json.loads(vocab_file.read_text())
    assert "model" in data


def test_load_vocab_round_trip(vocab_file, tokenizer):
    # tokenizer loaded from file should encode without error
    result = tokenizer.encode("a dog")
    assert len(result.ids) > 0


def test_special_tokens_present(tokenizer):
    for tok in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
        assert tokenizer.token_to_id(tok) is not None, f"{tok} missing from vocab"


def test_encode_length_equals_max_len(tokenizer):
    ids = encode(tokenizer, "a dog runs", max_len=10)
    assert len(ids) == 10


def test_encode_starts_with_bos(tokenizer):
    ids = encode(tokenizer, "a dog", max_len=10)
    assert ids[0] == tokenizer.token_to_id(BOS_TOKEN)


def test_encode_ends_with_eos_or_pad(tokenizer):
    # short caption: last real token before padding should be EOS
    ids = encode(tokenizer, "dog", max_len=10)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    assert eos_id in ids


def test_encode_pads_to_max_len(tokenizer):
    # one-word caption → lots of padding
    ids = encode(tokenizer, "dog", max_len=15)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    assert ids[-1] == pad_id
    assert len(ids) == 15


def test_encode_truncates_long_caption(tokenizer):
    long = " ".join(["dog"] * 50)
    ids = encode(tokenizer, long, max_len=8)
    assert len(ids) == 8
    # BOS still first, EOS must still be present despite truncation
    assert ids[0] == tokenizer.token_to_id(BOS_TOKEN)
    assert tokenizer.token_to_id(EOS_TOKEN) in ids


def test_unknown_token_handled(tokenizer):
    # a token that cannot possibly be in a 200-word vocab trained on 5 sentences
    ids = encode(tokenizer, "zxqwerty", max_len=10)
    # should not raise and should still produce a list of correct length
    assert len(ids) == 10


def test_load_vocab_missing_file():
    with pytest.raises(FileNotFoundError, match="Vocabulary not found"):
        load_vocab(Path("/nonexistent/vocab.json"))


def test_pad_token_is_id_zero(tokenizer):
    # The decoder's padding_idx and padding mask rely on this.
    assert tokenizer.token_to_id(PAD_TOKEN) == 0


def test_decode_round_trips_to_spaced_words(tokenizer):
    ids = tokenizer.encode("a dog runs in the park").ids
    assert tokenizer.decode(ids) == "a dog runs in the park"


def test_decode_skips_special_tokens(tokenizer):
    ids = encode(tokenizer, "a dog", max_len=10)
    assert tokenizer.decode(ids) == "a dog"


def test_build_vocab_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "er" / "vocab.json"
    build_vocab(_CORPUS, vocab_size=100, save_path=path)
    assert path.exists()


def test_special_ids_returns_pad_bos_eos(tokenizer):
    pad, bos, eos = special_ids(tokenizer)
    assert (pad, bos, eos) == tuple(
        tokenizer.token_to_id(t) for t in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)
    )


def test_special_ids_rejects_a_vocab_without_special_tokens():
    from tokenizers import Tokenizer, models
    bare = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    with pytest.raises(ValueError, match="no '<pad>' token"):
        special_ids(bare)


def test_encode_needs_room_for_bos_and_eos(tokenizer):
    with pytest.raises(ValueError):
        encode(tokenizer, "dog", max_len=1)


def test_group_pieces_merges_subwords():
    pieces = ["a" + WORD_END, "runn", "ing" + WORD_END, "dog" + WORD_END]
    assert group_pieces(pieces) == [("a", [0]), ("running", [1, 2]), ("dog", [3])]


def test_group_pieces_handles_truncated_word_and_unk():
    assert group_pieces(["ru", "nn"]) == [("runn", [0, 1])]
    assert group_pieces([UNK_TOKEN, "x" + WORD_END]) == [("?x", [0, 1])]
    assert group_pieces([]) == []
