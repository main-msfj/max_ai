"""Serializable configuration for SlidingWindowCompaction."""

from __future__ import annotations

from pydantic import Field

from ....core.compaction import CompactionConfig


class SlidingWindowCompactionConfig(CompactionConfig):
    max_turns: int = Field(
        default=10, ge=1, description="Turns kept; older ones leave the window.",
    )
