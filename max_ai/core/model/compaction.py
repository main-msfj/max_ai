"""Core models for context compaction."""

from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from ..messages import CoreMessage


class CompactionState(BaseModel):
    """Compaction bookkeeping kept in ``RunContext``; serializes with the session."""

    compactions: int = Field(default=0, description="Times the window was compacted.")
    archived_messages: int = Field(default=0, description="Messages that left the window.")
    state: dict[str, JsonValue] = Field(
        default_factory=dict, description="Strategy-owned state (e.g. its summary).",
    )


class CompactionResult(BaseModel):
    """What a compaction pass produced. The caller applies it to ``RunContext``."""

    changed: bool = False
    messages: list[CoreMessage] = Field(  # type: ignore[assignment]
        default_factory=list, description="The new live window.",
    )
    old_messages: list[CoreMessage] = Field(  # type: ignore[assignment]
        default_factory=list, description="Messages that left the window (memory input).",
    )
    state: dict[str, JsonValue] = Field(
        default_factory=dict, description="The strategy's new state.",
    )
    tokens_before: int = 0
    tokens_after: int = 0
    pruned_only: bool = Field(
        default=False, description="True when the cheap prune pass was enough.",
    )


class MessageGroup(BaseModel):
    """Atomic message group that should be kept or compacted as one unit."""

    messages: list[CoreMessage] = Field(default_factory=list)   # type: ignore[assignment]
    token_count: int = 0


__all__ = [
    "CompactionState",
    "CompactionResult",
    "MessageGroup",
]
