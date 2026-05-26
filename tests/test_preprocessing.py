import pytest
from src.data.preprocessing import clean_caption, clean_captions


def test_empty_string():
    assert clean_caption("") == ""


def test_already_clean():
    assert clean_caption("a dog runs fast") == "a dog runs fast"


def test_lowercase():
    assert clean_caption("The DOG") == "the dog"


def test_possessive():
    # this is the exact bug that str.replace("'s", "") would silently miss
    # when called on a pandas Series without regex=True
    assert clean_caption("dog's toy") == "dogs toy"


def test_possessive_end_of_string():
    assert clean_caption("the woman's") == "the womans"


def test_punctuation_stripped():
    assert clean_caption("hello, world!") == "hello world"


def test_punctuation_period():
    assert clean_caption("a dog. a cat.") == "a dog a cat"


def test_numeric_token_dropped():
    assert clean_caption("3 dogs run") == "dogs run"


def test_numeric_token_mid_sentence():
    assert clean_caption("he has 2 cats and 1 dog") == "he has cats and dog"


def test_alphanumeric_token_kept():
    # "a4" is not a purely numeric token — keep it
    assert "a4" in clean_caption("an a4 page")


def test_extra_whitespace_collapsed():
    assert clean_caption("a  big   dog") == "a big dog"


def test_leading_trailing_whitespace():
    assert clean_caption("  a dog  ") == "a dog"


def test_mixed_issues():
    # "2nd" is not a purely-numeric token (\b\d+\b won't match it), so it stays
    assert clean_caption("  The Dog's 2nd Run!!  ") == "the dogs 2nd run"


def test_clean_captions_batch():
    batch = ["Dog's toy", "Hello, World!", "3 cats"]
    result = clean_captions(batch)
    assert result == ["dogs toy", "hello world", "cats"]


def test_clean_captions_empty_list():
    assert clean_captions([]) == []
