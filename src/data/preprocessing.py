"""
Caption cleaning.

Everything here uses compiled ``re`` patterns. The original notebook used
``Series.str.replace("[^a-zA-Z]", "")``, which treats its first argument as a
*literal string*, so no caption was ever cleaned (see tasks/lessons.md, Bug 1).
"""
from __future__ import annotations

import re

# Apostrophes are deleted (not replaced by a space) so "dog's" -> "dogs".
_APOSTROPHE = re.compile(r"['’`]")
# Any other punctuation/symbol becomes a space so "black-and-white" keeps its
# words apart instead of fusing into "blackandwhite".
_PUNCTUATION = re.compile(r"[^\w\s]")
# Purely numeric tokens only: "2" goes, "2nd" and "a4" stay.
_NUMBER = re.compile(r"\b\d+\b")
_WHITESPACE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """Lowercase, drop punctuation and bare numbers, collapse whitespace."""
    text = text.lower()
    text = _APOSTROPHE.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _NUMBER.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def clean_captions(texts: list[str]) -> list[str]:
    """Vectorised convenience wrapper around :func:`clean_caption`."""
    return [clean_caption(t) for t in texts]
