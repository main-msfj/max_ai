"""Core contract for context compaction strategies."""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..core.messages import CoreMessage

if t.TYPE_CHECKING:
    from ..types.run_context import RunContext
    from ..types.stacks import PromptCtx


class CompactionResult(BaseModel):
    """Result of applying a compaction strategy to a run context."""

    changed: bool = False
    old_messages: list[CoreMessage] = Field(default_factory=list)
    recent_messages: list[CoreMessage] = Field(default_factory=list)
    old_token_count: int = 0
    recent_token_count: int = 0
    total_token_count: int = 0
    max_context_tokens: int = 0
    max_input_tokens: int = 0
    max_history_tokens: int = 0
    summary: str | None = None


class CoreCompaction(BaseModel, ABC):
    """Abstract base for strategies that keep a run within context budget.

    Concrete strategies may mutate ``ctx.messages`` and, later, selected
    prompt layers inside ``prompts``. They return a ``CompactionResult`` so
    callers can observe what changed and persist summaries when available.
    """

    @abstractmethod
    async def compact(
        self,
        *,
        ctx: "RunContext",
        prompts: "PromptCtx",
        max_context_tokens: int,
    ) -> CompactionResult:
        """Apply compaction to the provided runtime context."""
        ...
