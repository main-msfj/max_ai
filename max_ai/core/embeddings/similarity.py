"""Rank texts by cosine similarity of their vectors."""

from __future__ import annotations

import math
import typing as t

T = t.TypeVar("T")


def cosine_similarity(left: t.Sequence[float], right: t.Sequence[float]) -> float:
    """In [-1, 1]; 0.0 for empty, mismatched or zero vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def rank(
    query: t.Sequence[float],
    items: t.Sequence[T],
    vectors: t.Sequence[t.Sequence[float]],
    *,
    limit: int,
    min_score: float = 0.0,
) -> list[tuple[float, T]]:
    """The ``limit`` items most similar to ``query``, best first, above ``min_score``."""
    scored = [(cosine_similarity(query, vector), item) for item, vector in zip(items, vectors)]
    scored = [pair for pair in scored if pair[0] > min_score]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:limit]
