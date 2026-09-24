"""Serializable configuration for SummaryCompaction."""

from __future__ import annotations

from pydantic import Field

from ....core.compaction import CompactionConfig


class SummaryCompactionConfig(CompactionConfig):
    """Configuration options for ``SummaryCompaction``."""
    summary_max_tokens: int = Field(
        default=2000, gt=0,
        description="Output cap of each summary call; also reserved in the window.",
    )
    message_cap_tokens: int = Field(
        default=2000, gt=0,
        description="Each old message is cut to this before summarizing it.",
    )
