"""CoreCompaction.should_compact: when a strategy asks for compaction.

Numbers used below: window 10_000, output 1_000, prompt 1_000, 5% margin
(500) → capacity 7_500; at threshold 0.8 compaction starts above 6_000.
"""

from __future__ import annotations

from max_ai.base.compaction import CoreCompaction
from max_ai.core.compaction import CompactionResult
from max_ai.core.messages import UserMessage
from max_ai.types.chat_history import ChatHistory
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx

WINDOW, OUTPUT = 10_000, 1_000


class KeepAll(CoreCompaction):
    component_provider_override = "tests.KeepAll"

    async def _compact(self, blocks, *, state, budget_tokens, client):
        return CompactionResult(messages=[m for b in blocks for m in b.messages])


def prompts(tokens: int = 1_000) -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=tokens,
    )


def msg(tokens: int) -> UserMessage:
    return UserMessage(source="u", content="x", token_count=tokens)


def ctx(*tokens: int, history: tuple[int, ...] = ()) -> RunContext:
    return RunContext(
        messages=[msg(n) for n in tokens],
        message_history=ChatHistory(message_history=[msg(n) for n in history]),
    )


def test_below_and_above_the_threshold():
    strategy = KeepAll()
    assert not strategy.should_compact(ctx(3_000, 3_000), prompts(), WINDOW, OUTPUT)
    assert strategy.should_compact(ctx(3_000, 3_001), prompts(), WINDOW, OUTPUT)


def test_unknown_window_never_compacts():
    assert not KeepAll().should_compact(ctx(9_999_999), prompts(), 0, OUTPUT)


def test_message_history_counts_against_the_window():
    strategy = KeepAll()
    assert not strategy.should_compact(ctx(4_000), prompts(), WINDOW, OUTPUT)
    assert strategy.should_compact(ctx(4_000, history=(2_500,)), prompts(), WINDOW, OUTPUT)


def test_threshold_comes_from_the_strategy_not_global_settings():
    early, late = KeepAll(threshold=0.5), KeepAll(threshold=0.9)
    run = ctx(5_000)  # 50% of 7_500 = 3_750, 90% = 6_750
    assert early.should_compact(run, prompts(), WINDOW, OUTPUT)
    assert not late.should_compact(run, prompts(), WINDOW, OUTPUT)


def test_a_bigger_system_prompt_leaves_less_room():
    strategy = KeepAll()
    run = ctx(5_000)
    assert not strategy.should_compact(run, prompts(1_000), WINDOW, OUTPUT)
    # 10_000 - 3_000 - 1_000 - 500 = 5_500 capacity → threshold 4_400
    assert strategy.should_compact(run, prompts(3_000), WINDOW, OUTPUT)


def test_does_not_mutate_the_context():
    run = ctx(3_000, 3_001)
    before = run.model_dump_json()
    KeepAll().should_compact(run, prompts(), WINDOW, OUTPUT)
    assert run.model_dump_json() == before
