"""Ways of compacting a conversation. The contract is ``base.compaction.CoreCompaction``."""

from ...base.compaction import WindowCompaction as SlidingWindowCompaction
from .summary import SummaryCompaction, SummaryCompactionConfig

__all__ = [
    "SlidingWindowCompaction",
    "SummaryCompaction",
    "SummaryCompactionConfig",
]
