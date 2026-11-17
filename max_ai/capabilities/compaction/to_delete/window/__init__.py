"""Sliding window compaction: keep the last N turns, no LLM."""

from ._model import SlidingWindowCompactionConfig
from ._strategy import SlidingWindowCompaction

__all__ = ["SlidingWindowCompaction", "SlidingWindowCompactionConfig"]
