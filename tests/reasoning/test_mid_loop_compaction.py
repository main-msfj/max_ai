"""Tests for mid-loop compaction in both ReAct loops."""

from __future__ import annotations

import pytest
import typing as t

from max_ai.reasoning.react_simple import ReActLoop as SimpleReActLoop, ReActLoopState as SimpleLoopState
from max_ai.reasoning.react_planning import ReActLoopPlanning as PlanningReActLoop, ReActLoopPlanningState as PlanningLoopState
from max_ai.core.messages import AssistantMessage, ToolMessage, UserMessage
from max_ai.core.event_type import ReasoningCompleteEvent, CompactionEvent
from max_ai.core.models import ModelConfig
from max_ai.core.compaction import CompactionResult
from max_ai.base.compaction import CoreCompaction
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tool_call import ToolCallRecord
from max_ai.base.reasoning import BaseLoopState


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list, max_context_window: int = 0):
        self.model = "fake"
        self.config = ModelConfig(max_context_window=max_context_window)
        self._results = list(results)

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kw):
        return self._results.pop(0)

    def format_messages(self, ctx, prompts): return []
    def build_api_messages(self, messages): return []
    def build_tool_schema(self, tools): return []
    def normalize_usage_stats(self, usage): return Usage()
    async def complete(self, *a, **kw): raise NotImplementedError
    async def stream(self, *a, **kw): raise NotImplementedError


class FakeToolExecutor:
    def __init__(self):
        self.tools: dict = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            yield ToolMessage(
                source="tool",
                tool_call_id=record.id,
                tool_name=record.tool_name,
                success=True,
                content="ok",
            )


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


class FakeCompaction(CoreCompaction):
    """Compaction that records how many times it ran and trims ctx.messages to last N."""

    keep_last: int = 1
    call_count: int = 0

    async def compact(self, *, ctx, prompts, max_context_tokens, client) -> CompactionResult:
        self.call_count += 1
        old = ctx.messages[: -self.keep_last] if len(ctx.messages) > self.keep_last else []
        recent = ctx.messages[-self.keep_last:] if ctx.messages else []
        return CompactionResult(
            changed=bool(old),
            old_messages=old,
            recent_messages=recent,
            old_token_count=len(old) * 10,
            recent_token_count=len(recent) * 10,
            total_token_count=len(ctx.messages) * 10,
        )


class NeverCompaction(CoreCompaction):
    """Compaction that always says nothing needs compacting (should never run)."""

    async def compact(self, *, ctx, prompts, max_context_tokens, client) -> CompactionResult:
        raise AssertionError("compact() should not have been called")


# -------- HELPERS -------------------------------------------------------------
def make_result(content="done", tool_calls=None):
    from max_ai.core.messages import ToolCall
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason="stop" if not tool_calls else "tool_calls",
    )


def make_tool_call_result(call_id: str = "c1"):
    from max_ai.core.messages import ToolCall
    return make_result(
        content="",
        tool_calls=[ToolCall(id=call_id, tool_name="some_tool", parameters={})],
    )


def make_simple_loop(client, compaction=None, max_context_tokens=0):
    loop = SimpleReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(),
        middleware_chain=FakeMiddlewareChain(),
        compaction=compaction,
        max_context_tokens=max_context_tokens,
    )
    return loop


def make_planning_loop(client, compaction=None, max_context_tokens=0):
    loop = PlanningReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(),
        middleware_chain=FakeMiddlewareChain(),
        compaction=compaction,
        max_context_tokens=max_context_tokens,
    )
    return loop


@pytest.fixture
def ctx():
    return RunContext()


@pytest.fixture
def prompts():
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


async def collect(gen) -> list:
    return [ev async for ev in gen]


# -------- _should_compact unit tests ------------------------------------------
def test_should_compact_returns_false_without_compaction(ctx):
    """No compaction object → _should_compact always returns False."""
    client = FakeChatClient(results=[], max_context_window=1000)
    loop = make_simple_loop(client, compaction=None, max_context_tokens=1000)
    assert loop._should_compact(ctx) is False


def test_should_compact_returns_false_without_context_window(ctx):
    """max_context_tokens=0 → _should_compact always returns False."""
    loop = make_simple_loop(FakeChatClient(results=[]), compaction=FakeCompaction(), max_context_tokens=0)
    assert loop._should_compact(ctx) is False


def test_should_compact_returns_false_when_under_threshold(ctx):
    """Few messages → under threshold → no compaction needed."""
    loop = make_simple_loop(
        FakeChatClient(results=[], max_context_window=200_000),
        compaction=FakeCompaction(),
        max_context_tokens=200_000,
    )
    ctx.messages = [UserMessage(source="user", content="hi")]
    assert loop._should_compact(ctx) is False


def test_should_compact_returns_true_when_over_threshold(ctx):
    """Many large messages → over threshold → compaction needed."""
    loop = make_simple_loop(
        FakeChatClient(results=[], max_context_window=10_000),
        compaction=FakeCompaction(),
        max_context_tokens=10_000,
    )
    # threshold at 10k window ≈ 2000 tokens — 10 msgs * 300 words ≈ 3000 tokens
    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 10
    assert loop._should_compact(ctx) is True


# -------- _run_mid_loop_compaction unit tests ---------------------------------
@pytest.mark.asyncio
async def test_run_mid_loop_compaction_emits_event(ctx, prompts):
    """_run_mid_loop_compaction yields a CompactionEvent."""
    compaction = FakeCompaction(keep_last=1)
    loop = make_simple_loop(
        FakeChatClient(results=[], max_context_window=200_000),
        compaction=compaction,
        max_context_tokens=200_000,
    )
    ctx.messages = [
        UserMessage(source="user", content="msg1"),
        UserMessage(source="user", content="msg2"),
    ]

    events = [ev async for ev in loop._run_mid_loop_compaction(ctx, prompts)]

    assert len(events) == 1
    assert isinstance(events[0], CompactionEvent)
    assert events[0].phase == "end"
    assert events[0].changed is True


@pytest.mark.asyncio
async def test_run_mid_loop_compaction_trims_messages(ctx, prompts):
    """_run_mid_loop_compaction replaces ctx.messages with recent_messages."""
    compaction = FakeCompaction(keep_last=1)
    loop = make_simple_loop(
        FakeChatClient(results=[], max_context_window=200_000),
        compaction=compaction,
        max_context_tokens=200_000,
    )
    msg1 = UserMessage(source="user", content="old")
    msg2 = UserMessage(source="user", content="recent")
    ctx.messages = [msg1, msg2]

    [ev async for ev in loop._run_mid_loop_compaction(ctx, prompts)]

    assert len(ctx.messages) == 1
    assert ctx.messages[0].content == "recent"


@pytest.mark.asyncio
async def test_run_mid_loop_compaction_no_op_without_compactor(ctx, prompts):
    """Without a compactor, _run_mid_loop_compaction yields nothing."""
    loop = make_simple_loop(FakeChatClient(results=[]), compaction=None)
    ctx.messages = [UserMessage(source="user", content="x")]

    events = [ev async for ev in loop._run_mid_loop_compaction(ctx, prompts)]

    assert events == []
    assert len(ctx.messages) == 1  # untouched


# -------- Integration: simple ReActLoop ---------------------------------------
@pytest.mark.asyncio
async def test_simple_loop_no_compaction_event_when_not_configured(ctx, prompts):
    """Without compaction configured, no CompactionEvent is emitted."""
    client = FakeChatClient(results=[make_result("done")])
    loop = make_simple_loop(client)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=SimpleLoopState()))

    assert not any(isinstance(e, CompactionEvent) for e in events)


@pytest.mark.asyncio
async def test_simple_loop_no_compaction_when_under_threshold(ctx, prompts):
    """Compaction configured but context is small → compact() never called."""
    compaction = FakeCompaction()
    client = FakeChatClient(results=[make_result("done")], max_context_window=200_000)
    loop = make_simple_loop(client, compaction=compaction, max_context_tokens=200_000)

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=SimpleLoopState()))

    assert compaction.call_count == 0


@pytest.mark.asyncio
async def test_simple_loop_emits_compaction_event_when_over_threshold(ctx, prompts):
    """Loop emits CompactionEvent when context exceeds threshold before an iteration."""
    compaction = FakeCompaction(keep_last=1)
    client = FakeChatClient(results=[make_result("done")], max_context_window=10_000)
    loop = make_simple_loop(client, compaction=compaction, max_context_tokens=10_000)

    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 10

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=SimpleLoopState()))

    compaction_events = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compaction_events) >= 1
    assert compaction_events[0].phase == "end"
    assert compaction_events[0].changed is True


@pytest.mark.asyncio
async def test_simple_loop_compaction_trims_messages_before_llm(ctx, prompts):
    """After compaction fires, ctx.messages is trimmed before the LLM call."""
    compaction = FakeCompaction(keep_last=1)
    client = FakeChatClient(results=[make_result("done")], max_context_window=10_000)
    loop = make_simple_loop(client, compaction=compaction, max_context_tokens=10_000)

    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 10

    await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=SimpleLoopState()))

    assert compaction.call_count >= 1


@pytest.mark.asyncio
async def test_simple_loop_still_completes_after_compaction(ctx, prompts):
    """Loop reaches finish_reason='stop' even when compaction fires mid-loop."""
    compaction = FakeCompaction(keep_last=1)
    client = FakeChatClient(results=[make_result("final answer")], max_context_window=500)
    loop = make_simple_loop(client, compaction=compaction, max_context_tokens=500)

    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 5

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=SimpleLoopState()))

    complete = next(e for e in events if isinstance(e, ReasoningCompleteEvent))
    assert complete.finish_reason == "stop"


# -------- Integration: planning ReActLoop ------------------------------------
@pytest.mark.asyncio
async def test_planning_loop_emits_compaction_event(ctx, prompts):
    """Planning loop also emits CompactionEvent when threshold exceeded."""
    compaction = FakeCompaction(keep_last=1)
    client = FakeChatClient(results=[make_result("done")], max_context_window=10_000)
    loop = make_planning_loop(client, compaction=compaction, max_context_tokens=10_000)

    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 10

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=PlanningLoopState()))

    compaction_events = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compaction_events) >= 1
    assert compaction_events[0].changed is True


@pytest.mark.asyncio
async def test_planning_loop_completes_after_compaction(ctx, prompts):
    """Planning loop finishes correctly after mid-loop compaction."""
    compaction = FakeCompaction(keep_last=1)
    client = FakeChatClient(results=[make_result("answer")], max_context_window=500)
    loop = make_planning_loop(client, compaction=compaction, max_context_tokens=500)

    big_msg = UserMessage(source="user", content="word " * 300)
    ctx.messages = [big_msg] * 5

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=PlanningLoopState()))

    complete = next(e for e in events if isinstance(e, ReasoningCompleteEvent))
    assert complete.finish_reason == "stop"
