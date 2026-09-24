"""Compaction strategies. The contract is ``base.compaction.CoreCompaction``."""

from .summary import SummaryCompaction, SummaryCompactionConfig
from .window import SlidingWindowCompaction, SlidingWindowCompactionConfig

__all__ = [
    "SlidingWindowCompaction",
    "SlidingWindowCompactionConfig",
    "SummaryCompaction",
    "SummaryCompactionConfig",
]
