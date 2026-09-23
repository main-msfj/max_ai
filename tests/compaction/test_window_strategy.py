"""SlidingWindowCompaction: keep the last N turns, no LLM."""

from __future__ import annotations

from types import SimpleNamespace

from max_ai.capabilities.compaction import SlidingWindowCompaction
from max_ai.core.messages import (
    HARNESS_SOURCE,
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx

CLIENT = SimpleNamespace(generation_options={"max_tokens": 1_000})


def prompts() -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=1_000,
    )


def turn(n: int, tokens: int = 0, tool: bool = False) -> list:
    """One turn: question, optional tool call + result, answer."""
    msgs = [UserMessage(source="marvin", content=f"q{n}", token_count=tokens)]
    if tool:
        msgs += [
            AssistantMessage(source="a", content="", tool_calls=[ToolCall(id=f"c{n}", tool_name="bash")]),
            ToolMessage(source="t", content="ok", tool_call_id=f"c{n}", tool_name="bash", success=True),
        ]
    return msgs + [AssistantMessage(source="a", content=f"a{n}", token_count=tokens)]


def chat(*turns) -> RunContext:
    return RunContext(messages=[m for t in turns for m in t])


async def compact(strategy, ctx, window=10_000):
    return await strategy.compact(ctx=ctx, prompts=prompts(), max_context_tokens=window, client=CLIENT)


async def test_slides_by_turns_even_with_tokens_to_spare():
    ctx = chat(*(turn(n, tool=n % 2 == 0) for n in range(1, 6)))  # tiny, 5 turns
    before = list(ctx.messages)
    window = SlidingWindowCompaction(max_turns=3)
    assert window.should_compact(ctx, prompts(), 10_000, 1_000)
    result = await compact(window, ctx)
    assert not result.pruned_only and result.changed
    assert [m.text() for m in result.messages if isinstance(m, UserMessage)] == ["q3", "q4", "q5"]
    assert result.old_messages[0].text() == "q1"
    assert result.state == {} and window.render(result.state) is None  # no summary
    assert ctx.messages == before  # ctx untouched


async def test_within_the_turn_limit_nothing_changes():
    ctx = chat(turn(1), turn(2), turn(3))
    window = SlidingWindowCompaction(max_turns=3)
    assert not window.should_compact(ctx, prompts(), 10_000, 1_000)
    result = await compact(window, ctx)
    assert not result.changed and result.messages == ctx.messages


async def test_unknown_model_window_still_slides_by_turns():
    ctx = chat(*(turn(n) for n in range(1, 6)))
    window = SlidingWindowCompaction(max_turns=2)
    assert window.should_compact(ctx, prompts(), 0, 1_000)
    result = await compact(window, ctx, window=0)
    assert [m.text() for m in result.messages if isinstance(m, UserMessage)] == ["q4", "q5"]


async def test_a_huge_turn_also_trims_by_tokens():
    # 2 turns (under max_turns) but 8_000 tokens > 6_000 threshold.
    ctx = chat(turn(1, tokens=2_000), turn(2, tokens=2_000, tool=True))
    window = SlidingWindowCompaction(max_turns=10)
    assert window.should_compact(ctx, prompts(), 10_000, 1_000)
    result = await compact(window, ctx)
    assert result.tokens_after <= 3_000  # keep_ratio budget
    assert result.messages[-1].text() == "a2"


async def test_harness_messages_do_not_count_as_turns():
    harness = UserMessage(source=HARNESS_SOURCE, content="max iterations reached")
    ctx = RunContext(messages=[*turn(1), harness, *turn(2)])
    assert not SlidingWindowCompaction(max_turns=2).should_compact(ctx, prompts(), 10_000, 1_000)


def test_serializes_with_its_turn_limit():
    window = SlidingWindowCompaction(max_turns=4, threshold=0.7)
    restored = SlidingWindowCompaction.deserialize(window.serialize())
    assert (restored.config.max_turns, restored.config.threshold) == (4, 0.7)
