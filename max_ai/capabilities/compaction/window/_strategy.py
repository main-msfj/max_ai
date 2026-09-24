"""Keep the last N turns. No LLM, no summary: older turns just leave."""

from __future__ import annotations

from pydantic import JsonValue

from ....base.clients import CoreChatCompletionClient
from ....base.compaction import CoreCompaction
from ....core.compaction import (
    CompactionResult,
    MessageGroup,
    split_recent_messages,
    turn_starts,
)
from ._model import SlidingWindowCompactionConfig


class SlidingWindowCompaction(CoreCompaction):
    """Slide a window of the last ``max_turns`` turns over the conversation.

    A turn starts at a real user message. The window slides whenever there
    are more turns than ``max_turns`` (even with tokens to spare) and also
    when the token threshold is crossed: then it also drops the oldest
    blocks of the kept turns until they fit the budget. With an unknown
    model window only the turn count applies.
    """

    component_schema = SlidingWindowCompactionConfig
    component_provider_override = "maxai.compaction.SlidingWindowCompaction"

    def __init__(self, *, max_turns: int = 10, **kwargs) -> None:
        """Initialize ``SlidingWindowCompaction``.

Parameters
----------
max_turns : int
    Value supplied for ``max_turns``.
kwargs
    Value supplied for ``kwargs``."""
        super().__init__(**kwargs)
        self.config = SlidingWindowCompactionConfig(
            **self.config.model_dump(), max_turns=max_turns,
        )

    def _over_limit(
        self, blocks: list[MessageGroup], *, live_tokens: int, threshold: int,
    ) -> bool:
        """Perform the internal ``over limit`` operation for ``SlidingWindowCompaction``.

Parameters
----------
blocks : list[MessageGroup]
    Value supplied for ``blocks``.
live_tokens : int
    Value supplied for ``live_tokens``.
threshold : int
    Value supplied for ``threshold``."""
        too_many_turns = len(turn_starts(blocks)) > self.config.max_turns
        return too_many_turns or super()._over_limit(
            blocks, live_tokens=live_tokens, threshold=threshold,
        )

    async def _compact(
        self,
        blocks: list[MessageGroup],
        *,
        state: dict[str, JsonValue],
        budget_tokens: int,
        client: CoreChatCompletionClient,
    ) -> CompactionResult:
        """Perform the internal ``compact`` operation for ``SlidingWindowCompaction``.

Parameters
----------
blocks : list[MessageGroup]
    Value supplied for ``blocks``.
state : dict[str, JsonValue]
    Value supplied for ``state``.
budget_tokens : int
    Value supplied for ``budget_tokens``.
client : CoreChatCompletionClient
    Value supplied for ``client``."""
        starts = turn_starts(blocks)
        cut = starts[-self.config.max_turns] if len(starts) > self.config.max_turns else 0
        old, kept = blocks[:cut], blocks[cut:]

        old_messages = [m for block in old for m in block.messages]
        if budget_tokens > 0 and sum(block.token_count for block in kept) > budget_tokens:
            dropped, recent = split_recent_messages(
                kept, max_tokens=budget_tokens, min_keep_groups=self.config.min_keep_groups,
            )
            old_messages += dropped
        else:
            recent = [m for block in kept for m in block.messages]

        return CompactionResult(
            changed=bool(old_messages), messages=recent, old_messages=old_messages, state=state,
        )
