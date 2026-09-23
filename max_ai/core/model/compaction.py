"""Core models for context compaction."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field, JsonValue

from ..messages import AssistantMessage, CoreMessage


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


class CompactionConfig(BaseModel):
    """Config shared by every compaction strategy. Strategies extend it."""

    threshold: float = Field(
        default=0.8, gt=0, lt=1,
        description="Compact when live messages use this share of their capacity.",
    )
    keep_ratio: float = Field(
        default=0.4, gt=0, lt=1,
        description="Share of capacity kept as raw recent messages.",
    )
    min_keep_groups: int = Field(
        default=2, ge=1, description="Newest blocks always kept, budget or not.",
    )
    truncate_tool_outputs: bool = Field(
        default=True, description="Shrink old tool results/arguments before summarizing.",
    )
    tool_output_max_tokens: int = Field(default=500, gt=0)
    drop_harness_messages: bool = Field(
        default=True, description="Drop harness messages from previous turns.",
    )
    client: dict[str, t.Any] | None = Field(
        default=None,
        description="Serialized client for summaries/memory. None = the agent's client.",
    )


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

    @property
    def is_tool_group(self) -> bool:
        return any(
            isinstance(message, AssistantMessage) and bool(message.tool_calls)
            for message in self.messages
        )


__all__ = [
    "CompactionConfig",
    "CompactionState",
    "CompactionOutput",
    "MemoryFactUpdate",
    "MemoryMaintenanceOutput",
    "CompactionResult",
    "MessageGroup",
]
