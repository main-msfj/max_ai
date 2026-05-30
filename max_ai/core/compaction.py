"""Core models for context compaction."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .messages import AssistantMessage, CoreMessage


class CompactionOutput(BaseModel):
    """Structured summary produced by future summary-based compaction."""

    summary: str = Field(
        default="",
        description="Quick summary of what is going on in the conversation.",
    )
    objective: list[str] = Field(
        default_factory=list,
        description="Current user or agent objectives that still matter.",
    )
    pending: list[str] = Field(
        default_factory=list,
        description="Open questions, blockers, approvals, or unfinished work.",
    )
    successfully_done: list[str] = Field(
        default_factory=list,
        description="Work already completed successfully.",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Important decisions, constraints, or preferences established.",
    )
    important_context: list[str] = Field(
        default_factory=list,
        description="Facts that should be preserved for future turns.",
    )


class MemoryFactUpdate(BaseModel):
    """Memory fact to create or update after compaction."""

    key: str = Field(..., description="Stable memory key to create or update.")
    category: str = Field(..., description="Memory category.")
    content: str = Field(..., description="Updated durable memory content.")
    reason: str | None = Field(default=None, description="Why this update is justified.")


class MemoryMaintenanceOutput(BaseModel):
    """Structured memory maintenance produced during compaction."""

    updates: list[MemoryFactUpdate] = Field(
        default_factory=list,
        description="Memory facts that should be created or overwritten.",
    )


class CompactionResult(BaseModel):
    """Result of applying a compaction strategy to a run context."""

    changed: bool = False
    old_messages: list[CoreMessage] = Field(default_factory=list)   # type: ignore[assignment]
    recent_messages: list[CoreMessage] = Field(default_factory=list)    # type: ignore[assignment]
    old_token_count: int = 0
    recent_token_count: int = 0
    total_token_count: int = 0
    summary: str | None = None


class MessageGroup(BaseModel):
    """Atomic message group that should be kept or compacted as one unit."""

    messages: list[CoreMessage] = Field(default_factory=list)   # type: ignore[assignment]
    token_count: int = 0

    @property
    def is_tool_group(self) -> bool:
        return any(
            isinstance(message, AssistantMessage) and bool(message.tool_calls)
            for message in self.messages
        )


__all__ = [
    "CompactionOutput",
    "MemoryFactUpdate",
    "MemoryMaintenanceOutput",
    "CompactionResult",
    "MessageGroup",
]
