"""A text as the words a search compares."""

from __future__ import annotations

import re
from collections.abc import Collection

_WORD = re.compile(r"[^\W_]+")  # letters and digits; "_" splits exa_search


def words(text: str, stopwords: Collection[str] = frozenset()) -> list[str]:
    """The lowercase words of ``text``, without ``stopwords``."""
    return [word for word in _WORD.findall(text.lower()) if word not in stopwords]
