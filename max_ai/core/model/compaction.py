"""Core models for context compaction."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..messages import CoreMessage


class CompactionState(BaseModel):
    """Compaction bookkeeping kept in ``RunContext``; serializes with the session."""

    compactions: int = Field(default=0, description="Times the window was compacted.")


class MessageGroup(BaseModel):
    """Atomic message group that should be kept or compacted as one unit."""

    messages: list[CoreMessage] = Field(default_factory=list)   # type: ignore[assignment]
    token_count: int = 0


__all__ = [
    "CompactionState",
    "MessageGroup",
]
