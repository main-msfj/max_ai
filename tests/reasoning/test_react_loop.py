"""ReActLoop integration tests with fake client/executor/middleware."""

from __future__ import annotations

import asyncio
import typing as t
import pytest

from max_ai.base.reasoning import BaseLoopState
from max_ai.reasoning.react_planning import (
    ReActLoopPlanning as ReActLoop,
    ReActLoopPlanningState as ReActLoopState,
)

from max_ai.base.clients import CoreChatCompletionClient
from max_ai.core.messages import AssistantMessage, ToolMessage, ToolCall
from max_ai.core.event_type import (
    ReasoningIterationEvent,
    ReasoningCompleteEvent,
    ToolApprovalEvent,
    ErrorEvent,
)
from max_ai.core.models import ModelConfig
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.errors.client import ClientError


# -------- FAKES -----------------------------------------------------------
class FakeChatClient(CoreChatCompletionClient):
    """Bypasses run() pipeline; returns scripted results from a queue."""

    def __init__(self, results=None, raise_first=None):
        self.model = "fake"
        self.config = ModelConfig()
        self._results: list[ChatCompletionResult] = list(results or [])
        self._raise_first = raise_first

    async def run(
        self, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs
    ):
        if self._raise_first is not None:
            err = self._raise_first
            self._raise_first = None
            raise err
        return self._results.pop(0)

    # Abstracts — never called because run() is overridden.
    def format_messages(self, ctx, prompts):
        return []

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    def normalize_usage_stats(self, usage):
        return Usage()

    async def complete(self, *a, **kw):
        raise NotImplementedError

    async def stream(self, *a, **kw):
        raise NotImplementedError
        yield


class FakeToolExecutor:
    def __init__(self, tools=None, scripted=None):
        self.tools: dict = {
            getattr(t, "name", f"t{i}"): t for i, t in enumerate(tools or [])
        }
        self._scripted: list[list] = list(scripted or [])
        self.calls: list[list] = []

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        self.calls.append(list(records))
        items = self._scripted.pop(0) if self._scripted else []
        for item in items:
            yield item


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, func, metadata=None):
        async for chunk in func(ctx):
            yield chunk


# -------- HELPERS -----------------------------------------------------------
def make_result(content="", tool_calls=None, finish_reason="stop"):
    return ChatCompletionResult(
        message=AssistantMessage(
            source="fake",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=Usage(llm_calls=1, attempts_to_call_api=1),
        model="fake",
        finish_reason=finish_reason,
    )


def make_loop(client, executor, max_iter=3, max_retries=3, chain=None):
    loop = ReActLoop(
        max_loop_iterations=max_iter,
        max_connection_retries=max_retries,
    )
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=executor,
        middleware_chain=chain or FakeMiddlewareChain(),
    )
    return loop


@pytest.fixture
def ctx():
    return RunContext()


@pytest.fixture
def prompts():
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


async def collect(gen):
    return [ev async for ev in gen]


# -------- TESTS -----------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_no_tools(ctx, prompts):
    """LLM responds without tools → 1 iteration, finish_reason=stop."""
    client = FakeChatClient(results=[make_result(content="hello world")])
    loop = make_loop(client, FakeToolExecutor())
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        )
    )

    assert state.iteration == 1
    assert state.finish_reason == "stop"
    assert len(ctx.messages) == 1
    assert ctx.messages[0].text() == "hello world"
    assert isinstance(events[0], ReasoningIterationEvent)
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "stop"
    assert events[-1].total_iterations == 1


@pytest.mark.asyncio
async def test_tool_path_two_iterations(ctx, prompts):
    """LLM → tool → LLM → final. ToolCallRecord registered on ctx."""
    tc = ToolCall(id="call_1", tool_name="ping", parameters={"x": 1})
    client = FakeChatClient(
        results=[
            make_result(tool_calls=[tc], finish_reason="tool_calls"),
            make_result(content="done"),
        ]
    )
    tool_msg = ToolMessage.success_message(
        tool_call_id="call_1",
        tool_name="ping",
        content="pong",
        source="exec",
    )
    executor = FakeToolExecutor(scripted=[[tool_msg]])
    loop = make_loop(client, executor)
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        )
    )

    assert state.iteration == 2
    assert state.finish_reason == "stop"
    assert state.tool_calls == 1
    assert len(ctx.messages) == 3  # assistant, tool, assistant
    assert ctx.tool_state.has("call_1")
    assert ctx.tool_state.get("call_1").tool_name == "ping"
    assert len(executor.calls) == 1
    assert executor.calls[0][0].id == "call_1"


@pytest.mark.asyncio
async def test_approval_pause_from_executor(ctx, prompts):
    """Executor emits ToolApprovalEvent → loop ends with approval_needed."""
    tc = ToolCall(id="call_1", tool_name="dangerous", parameters={})
    client = FakeChatClient(
        results=[
            make_result(tool_calls=[tc], finish_reason="tool_calls"),
        ]
    )
    approval = ToolApprovalEvent(
        source="exec",
        tool_call_id="call_1",
        tool_name="dangerous",
        parameters={},
    )
    executor = FakeToolExecutor(scripted=[[approval]])
    loop = make_loop(client, executor)
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        )
    )

    assert state.finish_reason == "approval_needed"
    assert state.iteration == 1
    assert any(isinstance(e, ToolApprovalEvent) for e in events)
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "approval_needed"


@pytest.mark.asyncio
async def test_max_iterations_exceeded(ctx, prompts):
    """LLM keeps producing tool calls → exhausts max_loop_iterations."""

    def loop_result():
        return make_result(
            tool_calls=[ToolCall(tool_name="ping", parameters={})],
            finish_reason="tool_calls",
        )

    client = FakeChatClient(results=[loop_result() for _ in range(5)])
    executor = FakeToolExecutor(
        scripted=[
            [
                ToolMessage.success_message(
                    tool_call_id=loop_result().message.tool_calls[0].id,
                    tool_name="ping",
                    content="ok",
                    source="exec",
                )
            ]
            for _ in range(5)
        ]
    )
    loop = make_loop(client, executor, max_iter=2)
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        )
    )

    assert state.iteration == 2
    assert state.finish_reason == "max_iterations_exceeded"


@pytest.mark.asyncio
async def test_no_result(ctx, prompts):
    """Middleware produces no result → finish_reason=no_result."""

    class EmptyChain(FakeMiddlewareChain):
        async def execute(self, action, ctx, data, func, metadata=None):
            await func(ctx)
            return
            yield  # pragma: no cover

    client = FakeChatClient(results=[make_result(content="ignored")])
    loop = make_loop(client, FakeToolExecutor(), chain=EmptyChain())
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(
        ctx=ctx, prompts=prompts, loop_state=state,
    ))

    assert state.finish_reason == "no_result"
    assert events[-1].finish_reason == "no_result"


@pytest.mark.asyncio
async def test_cancellation(ctx, prompts):
    """Cancelled token at iteration start → CancelledError."""

    class Cancelled:
        def is_cancelled(self):
            return True

        def link_future(self, fut):
            pass

    client = FakeChatClient(results=[make_result(content="never")])
    loop = make_loop(client, FakeToolExecutor())
    state = ReActLoopState()

    with pytest.raises(asyncio.CancelledError):
        await collect(
            loop.execute_reasoning_loop(
                ctx=ctx,
                prompts=prompts,
                loop_state=state,
                cancellation_token=Cancelled(),
            )
        )


@pytest.mark.asyncio
async def test_retry_on_transient_error(ctx, prompts, monkeypatch):
    """Transient ClientError on first attempt → retry succeeds."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *a, **kw: real_sleep(0))

    client = FakeChatClient(
        results=[make_result(content="recovered")],
        raise_first=ClientError.rate_limit_exceeded(),
    )
    loop = make_loop(client, FakeToolExecutor(), max_retries=2)
    state = ReActLoopState()

    await collect(
        loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        )
    )

    assert state.finish_reason == "stop"
    assert ctx.messages[0].text() == "recovered"


@pytest.mark.asyncio
async def test_non_transient_error_yields_error_event(ctx, prompts):
    """Non-transient ClientError → ErrorEvent yielded, exception raised."""
    client = FakeChatClient(
        results=[],
        raise_first=ClientError.authentication_failed("bad key"),
    )
    loop = make_loop(client, FakeToolExecutor(), max_retries=2)
    state = ReActLoopState()

    with pytest.raises(ClientError):
        events = []
        async for ev in loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=prompts,
            loop_state=state,
        ):
            events.append(ev)

    assert any(isinstance(e, ErrorEvent) for e in events)
