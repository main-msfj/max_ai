"""Keyword ranking (BM25) and the fusion of several rankings (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence

_WORD = re.compile(r"[^\W_]+")  # letters and digits; "_" splits exa_search


def words(text: str, stopwords: Collection[str] = frozenset()) -> list[str]:
    """The lowercase words of ``text``, without ``stopwords``."""
    return [word for word in _WORD.findall(text.lower()) if word not in stopwords]


def bm25(
    query: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """One score per doc: rare query words count more, long docs a bit less."""
    if not docs or not query:
        return [0.0] * len(docs)
    total = len(docs)
    average = sum(len(doc) for doc in docs) / total or 1.0
    seen_in = Counter(word for doc in docs for word in set(doc))
    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for word in set(query):
            if word not in counts:
                continue
            rarity = math.log(1 + (total - seen_in[word] + 0.5) / (seen_in[word] + 0.5))
            tf = counts[word]
            score += rarity * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / average))
        scores.append(score)
    return scores


def fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: only positions count, so scores of
    different scales (BM25, cosine) never have to be compared."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, key in enumerate(ranking, start=1):
            scores[key] += 1 / (k + position)
    return sorted(scores, key=lambda key: (-scores[key], key))
