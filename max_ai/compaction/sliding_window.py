from __future__ import annotations

import typing as t

from pydantic import Field

from ..base.compaction import CompactionResult, CoreCompaction
from ..core.compaction import (
    TokenBudgetStrategy,
    group_atomic_messages,
    split_recent_messages,
)

if t.TYPE_CHECKING:
    from ..types.run_context import RunContext
    from ..types.stacks import PromptCtx


class SlidingWindowCompaction(CoreCompaction):
    """Keep the most recent raw history that fits the history token budget.

    This v1 strategy does not create a summary. It only moves older
    messages out of the active transcript and keeps the newest atomic
    message groups raw.
    """

    token_strategy: TokenBudgetStrategy = Field(default_factory=TokenBudgetStrategy)

    async def compact(
        self,
        *,
        ctx: "RunContext",
        prompts: "PromptCtx",
        max_context_tokens: int,
    ) -> CompactionResult:
        del prompts  # Reserved for summary-layer updates in the next phase.

        budget = self.token_strategy.build_budget(
            max_context_tokens=max_context_tokens
        )
        total_tokens = self.token_strategy.count_messages(ctx.messages)

        if total_tokens <= budget.max_history_tokens:
            return CompactionResult(
                changed=False,
                recent_messages=list(ctx.messages),
                recent_token_count=total_tokens,
                total_token_count=total_tokens,
                max_context_tokens=budget.max_context_tokens,
                max_input_tokens=budget.max_input_tokens,
                max_history_tokens=budget.max_history_tokens,
            )

        groups = group_atomic_messages(ctx.messages, self.token_strategy)
        old_messages, recent_messages = split_recent_messages(
            groups,
            max_history_tokens=budget.max_history_tokens,
        )

        if old_messages:
            ctx.messages = recent_messages

        return CompactionResult(
            changed=bool(old_messages),
            old_messages=old_messages,
            recent_messages=recent_messages,
            old_token_count=self.token_strategy.count_messages(old_messages),
            recent_token_count=self.token_strategy.count_messages(recent_messages),
            total_token_count=total_tokens,
            max_context_tokens=budget.max_context_tokens,
            max_input_tokens=budget.max_input_tokens,
            max_history_tokens=budget.max_history_tokens,
        )
