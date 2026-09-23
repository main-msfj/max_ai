"""CoreCompaction.compact: the pipeline every strategy runs through.

Numbers: window 10_000, output 1_000, prompt 1_000, 5% margin → capacity
7_500; threshold (0.8) 6_000; budget kept after compacting (0.4) 3_000.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from max_ai.base.compaction import CoreCompaction
from max_ai.core.compaction import CompactionResult, split_recent_messages
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.errors.compaction import CompactionError
from max_ai.types.chat_history import ChatHistory
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.tool_call import ToolCallRecord

WINDOW = 10_000
AGENT_CLIENT = SimpleNamespace(name="agent", generation_options={"max_tokens": 1_000})
BIG = "línea de log con bastante texto adentro\n" * 400  # ~4k tokens


class DropOld(CoreCompaction):
    """Keeps the newest blocks that fit the budget; records every call."""

    component_provider_override = "tests.DropOld"

    def __init__(self, **kwargs):
        super().__init__(min_keep_groups=2, keep_ratio=0.4, **kwargs)
        self.calls: list[dict] = []
        self.memory_calls: list[list] = []

    async def _compact(self, blocks, *, state, budget_tokens, client):
        self.calls.append({"budget": budget_tokens, "client": client, "blocks": len(blocks)})
        old, recent = split_recent_messages(blocks, max_tokens=budget_tokens, min_keep_groups=2)
        state["times"] = state.get("times", 0) + 1  # mutating the copy is allowed
        return CompactionResult(changed=True, messages=recent, old_messages=old, state=state)

    async def _update_memory(self, old_messages, *, memory, client):
        self.memory_calls.append(old_messages)
        if memory == "broken":
            raise RuntimeError("database down")


def prompts() -> PromptCtx:
    return PromptCtx.model_construct(
        stack=None, variables={}, rendered_layers={}, layer_usage={}, prompt_tokens=1_000,
    )


def user(tokens: int = 0, text: str = "x") -> UserMessage:
    return UserMessage(source="marvin", content=text, token_count=tokens)


def answer(tokens: int = 0) -> AssistantMessage:
    return AssistantMessage(source="a", content="ok", token_count=tokens)


def tool_block(call_id: str, output: str) -> list:
    return [
        AssistantMessage(source="a", content="", tool_calls=[ToolCall(id=call_id, tool_name="bash")]),
        ToolMessage(source="t", content=output, tool_call_id=call_id, tool_name="bash", success=True),
    ]


def long_chat(turns: int = 8) -> list:
    """Each turn: a 500-token question and a 500-token answer → 1_000 per turn."""
    return [m for _ in range(turns) for m in (user(500), answer(500))]


async def run(strategy, ctx, memory=None):
    return await strategy.compact(
        ctx=ctx, prompts=prompts(), max_context_tokens=WINDOW, client=AGENT_CLIENT, memory=memory,
    )


async def test_pruning_alone_is_enough_and_the_strategy_is_not_called():
    ctx = RunContext(messages=[user(1_500), *tool_block("c1", BIG), answer(), user(1_500), answer()])
    strategy = DropOld()
    result = await run(strategy, ctx)
    assert result.pruned_only and result.changed
    assert strategy.calls == []  # no LLM spent
    assert result.tokens_before > 6_000 >= result.tokens_after
    assert ctx.messages[2].text() == BIG  # ctx untouched; the loop applies the result


async def test_strategy_runs_when_pruning_is_not_enough():
    ctx = RunContext(messages=long_chat(8))  # 8_000 tokens, nothing to prune
    ctx.compaction.state["times"] = 1
    strategy = DropOld()
    result = await run(strategy, ctx)
    assert len(strategy.calls) == 1 and strategy.calls[0]["budget"] == 3_000
    assert not result.pruned_only and result.changed
    assert result.tokens_before == 8_000 and result.tokens_after <= 3_000
    assert len(result.messages) + len(result.old_messages) == 16
    assert result.state == {"times": 2}
    assert ctx.compaction.state == {"times": 1} and len(ctx.messages) == 16


async def test_history_takes_budget_away_from_messages():
    ctx = RunContext(
        messages=long_chat(6),  # 6_000 + 1_000 history = 7_000 > 6_000
        message_history=ChatHistory(message_history=[user(1_000)]),
    )
    strategy = DropOld()
    result = await run(strategy, ctx)
    assert strategy.calls[0]["budget"] == 2_000
    assert result.tokens_after <= 3_000  # history + kept messages


async def test_strategy_client_wins_over_the_agent_client():
    cheap = SimpleNamespace(name="cheap")
    default, own = DropOld(), DropOld(client=cheap)
    await run(default, RunContext(messages=long_chat(8)))
    await run(own, RunContext(messages=long_chat(8)))
    assert default.calls[0]["client"] is AGENT_CLIENT
    assert own.calls[0]["client"] is cheap


async def test_a_strategy_that_splits_a_block_is_rejected():
    class Broken(DropOld):
        async def _compact(self, blocks, *, state, budget_tokens, client):
            flat = [m for b in blocks for m in b.messages]
            return CompactionResult(messages=flat[1:])  # drops the call, keeps its result

    ctx = RunContext(messages=[*tool_block("c1", "ok"), *long_chat(8)])
    with pytest.raises(CompactionError, match="answers tool call 'c1'"):
        await run(Broken(), ctx)


async def test_memory_receives_what_left_the_window_and_failures_do_not_break_it():
    strategy = DropOld()
    result = await run(strategy, RunContext(messages=long_chat(8)), memory="memory")
    assert strategy.memory_calls == [result.old_messages]

    broken = DropOld()
    result = await run(broken, RunContext(messages=long_chat(8)), memory="broken")
    assert result.changed and len(broken.memory_calls) == 1  # logged, not raised


async def test_no_memory_connected_is_skipped():
    strategy = DropOld()
    await run(strategy, RunContext(messages=long_chat(8)))
    assert strategy.memory_calls == []


async def test_pending_tool_calls_block_compaction():
    ctx = RunContext(messages=long_chat(8))
    ctx.tool_state.add(ToolCallRecord(id="p1", tool_name="send_email"))  # awaiting approval
    strategy = DropOld()
    result = await run(strategy, ctx)
    assert not result.changed and strategy.calls == []
    assert result.messages == ctx.messages
