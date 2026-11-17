"""Ways of compacting a conversation. The contract is ``base.compaction.CoreCompaction``."""

from ...base.compaction import WindowCompaction as SlidingWindowCompaction
from ...base.compaction import WindowCompactionConfig as SlidingWindowCompactionConfig
from .summary import SummaryCompaction, SummaryCompactionConfig

__all__ = [
    "SlidingWindowCompaction",
    "SlidingWindowCompactionConfig",
    "SummaryCompaction",
    "SummaryCompactionConfig",
]
