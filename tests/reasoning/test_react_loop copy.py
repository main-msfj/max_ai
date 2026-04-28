"""
Unit tests for the concrete ReAct loop.

Covers the loop-level behavior that sits on top of BaseReasoning:
  - final answer without tools
  - full tool cycle and transcript mutation
  - approval pause behavior
  - max-iteration termination
  - streaming tool cycle
"""

from __future__ import annotations

import typing as t

import pytest

from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.tool_executor import ToolExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.core.event_type import (
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.core.models import ModelConfig
from max_ai.middleware.chain import MiddlewareChain
from max_ai.reasoning.react import ReActLoop, ReActLoopState
from max_ai.termination import CancellationToken
from max_ai.types.completions import (
    ChatCompletionChunk,
    ChatCompletionResult,
    Usage,
)
from max_ai.types.middleware import MiddlewareCtx
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import CoreToolParameters, ToolApprovalMode
from max_ai.validators.stacks import build_default_stack

MiddlewareCtx.model_rebuild(_types_namespace={"RunContext": RunContext})


class ScriptedClient(CoreChatCompletionClient):
    """Test client that returns pre-scripted complete or stream outputs."""

    def __init__(
        self,
        *,
        complete_results: list[ChatCompletionResult] | None = None,
        stream_results: list[list[ChatCompletionChunk]] | None = None,
    ) -> None:
        super().__init__(model="scripted-model", config=ModelConfig())
        self._complete_results = list(complete_results or [])
        self._stream_results = [list(chunks) for chunks in (stream_results or [])]
        self.complete_calls = 0
        self.stream_calls = 0

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def format_messages(self, ctx: RunContext, prompts: PromptCtx):
        return list(ctx.messages)

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools: list[CoreTool]):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        if not self._complete_results:
            raise AssertionError("No scripted complete result left.")
        self.complete_calls += 1
        return self._complete_results.pop(0)

    async def stream(self, messages, tools, output_format, **kwargs):
        if not self._stream_results:
            raise AssertionError("No scripted stream result left.")
        self.stream_calls += 1
        for chunk in self._stream_results.pop(0):
            yield chunk


class MockTool(CoreTool):
    """Simple tool with controllable output and approval mode."""

    def __init__(
        self,
        *,
        name: str = "mock_tool",
        approval_mode: ToolApprovalMode = ToolApprovalMode.AUTO_APPROVED,
        returns: t.Any = None,
    ) -> None:
        super().__init__(
            name=name,
            description="Mock tool for ReAct loop tests",
            approval_mode=approval_mode,
            timeout_seconds=10,
        )
        self._returns = returns
        self.execute_called = False

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        return CoreToolParameters(is_tool_valid=True, msg_error=None)

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.execute_called = True
        return ToolResult.success_result(tool_call_id=tool_request.id, result=self._returns)


def make_prompt_ctx() -> PromptCtx:
    stack = build_default_stack()
    return PromptCtx(
        stack=stack,
        variables={},
        rendered_layers={type(layer): "" for layer in stack},
    )


def make_ctx(user_text: str = "hello") -> RunContext:
    return RunContext(
        session_id="test-session",
        messages=[UserMessage(source="user", content=user_text)],
    )


def make_result(
    *,
    content: str,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
    usage: Usage | None = None,
) -> ChatCompletionResult:
    return ChatCompletionResult(
        message=AssistantMessage(
            source="scripted-model",
            content=content,
            tool_calls=tool_calls or [],
        ),
        usage=usage or Usage(llm_calls=1, attempts_to_call_api=1, tokens_input=10),
        model="scripted-model",
        finish_reason=finish_reason,
    )


async def collect(gen) -> list[t.Any]:
    return [item async for item in gen]


@pytest.mark.asyncio
async def test_react_loop_final_answer_stops_after_one_iteration():
    client = ScriptedClient(
        complete_results=[make_result(content="Final answer.", finish_reason="stop")]
    )
    loop = ReActLoop(
        name="test-agent",
        client=client,
        tool_executor=ToolExecutor(tools=[], agent_name="test-agent"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=3,
    )
    ctx = make_ctx()
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx, make_prompt_ctx(), state))

    assert isinstance(events[0], ReasoningIterationEvent)
    assert isinstance(events[1], ModelResponseEvent)
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "stop"

    assert len(ctx.messages) == 2
    assert isinstance(ctx.messages[-1], AssistantMessage)
    assert ctx.messages[-1].text() == "Final answer."
    assert state.iteration == 1
    assert state.llm_calls == 1


@pytest.mark.asyncio
async def test_react_loop_runs_tool_then_calls_llm_again():
    tool = MockTool(name="get_number", returns="7")
    client = ScriptedClient(
        complete_results=[
            make_result(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", tool_name="get_number", parameters={})
                ],
                finish_reason="tool_calls",
            ),
            make_result(content="The answer is 7.", finish_reason="stop"),
        ]
    )
    loop = ReActLoop(
        name="test-agent",
        client=client,
        tool_executor=ToolExecutor(tools=[tool], agent_name="test-agent"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=3,
    )
    ctx = make_ctx("Use the tool.")
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx, make_prompt_ctx(), state))

    assert sum(isinstance(e, ReasoningIterationEvent) for e in events) == 2
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolCallResponseEvent) for e in events)
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "stop"

    assert tool.execute_called is True
    assert len(ctx.messages) == 4
    assert isinstance(ctx.messages[1], AssistantMessage)
    assert isinstance(ctx.messages[2], ToolMessage)
    assert isinstance(ctx.messages[3], AssistantMessage)
    assert ctx.messages[2].content == "7"
    assert ctx.messages[3].text() == "The answer is 7."

    record = ctx.tool_state.get("call_1")
    assert record is not None
    assert record.is_consumed
    assert record.result is not None
    assert record.result.result == "7"

    assert state.iteration == 2
    assert state.tool_calls == 1
    assert state.llm_calls == 2


@pytest.mark.asyncio
async def test_react_loop_pauses_when_tool_needs_approval():
    tool = MockTool(
        name="delete_thing",
        approval_mode=ToolApprovalMode.ASK_APPROVED,
        returns="done",
    )
    client = ScriptedClient(
        complete_results=[
            make_result(
                content="",
                tool_calls=[
                    ToolCall(id="call_approval", tool_name="delete_thing", parameters={})
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    loop = ReActLoop(
        name="test-agent",
        client=client,
        tool_executor=ToolExecutor(tools=[tool], agent_name="test-agent"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=3,
    )
    ctx = make_ctx("Delete it.")
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx, make_prompt_ctx(), state))

    assert any(isinstance(e, ToolApprovalEvent) for e in events)
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "approval_needed"

    assert tool.execute_called is False
    assert len(ctx.messages) == 2
    assert isinstance(ctx.messages[-1], AssistantMessage)

    record = ctx.tool_state.get("call_approval")
    assert record is not None
    assert record.is_pending_approval
    assert state.tool_calls == 0


@pytest.mark.asyncio
async def test_react_loop_reports_max_iterations_exceeded():
    tool = MockTool(name="loop_tool", returns="intermediate")
    client = ScriptedClient(
        complete_results=[
            make_result(
                content="",
                tool_calls=[ToolCall(id="call_loop", tool_name="loop_tool", parameters={})],
                finish_reason="tool_calls",
            )
        ]
    )
    loop = ReActLoop(
        name="test-agent",
        client=client,
        tool_executor=ToolExecutor(tools=[tool], agent_name="test-agent"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=1,
    )
    ctx = make_ctx("Keep looping.")
    state = ReActLoopState()

    events = await collect(loop.execute_reasoning_loop(ctx, make_prompt_ctx(), state))

    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "max_iterations_exceeded"
    assert len(ctx.messages) == 3
    assert isinstance(ctx.messages[-1], ToolMessage)
    assert state.iteration == 1
    assert state.tool_calls == 1


@pytest.mark.asyncio
async def test_react_loop_streaming_cycle_merges_tool_call_and_finishes():
    tool = MockTool(name="get_weather", returns="sunny")
    client = ScriptedClient(
        stream_results=[
            [
                ChatCompletionChunk(
                    content="",
                    thinking=None,
                    is_complete=False,
                    tool_call_chunk={
                        "id": "call_stream",
                        "function": {"name": "get_weather", "arguments": '{"city":"'},
                    },
                ),
                ChatCompletionChunk(
                    content="",
                    thinking=None,
                    is_complete=False,
                    tool_call_chunk={
                        "id": "call_stream",
                        "function": {"arguments": 'Paris"}'},
                    },
                ),
                ChatCompletionChunk(
                    content="",
                    thinking=None,
                    is_complete=True,
                    usage=Usage(llm_calls=1, attempts_to_call_api=1, tokens_input=5),
                ),
            ],
            [
                ChatCompletionChunk(
                    content="Weather is sunny.",
                    thinking=None,
                    is_complete=False,
                ),
                ChatCompletionChunk(
                    content="",
                    thinking=None,
                    is_complete=True,
                    usage=Usage(llm_calls=1, attempts_to_call_api=1, tokens_input=4),
                ),
            ],
        ]
    )
    loop = ReActLoop(
        name="test-agent",
        client=client,
        tool_executor=ToolExecutor(tools=[tool], agent_name="test-agent"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=3,
    )
    ctx = make_ctx("Check the weather.")
    state = ReActLoopState()

    events = await collect(
        loop.execute_reasoning_loop(
            ctx,
            make_prompt_ctx(),
            state,
            stream_tokens=True,
        )
    )

    stream_events = [e for e in events if isinstance(e, ModelStreamChunkEvent)]
    assert stream_events
    assert stream_events[-1].is_final is True
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert events[-1].finish_reason == "stop"

    record = ctx.tool_state.get("call_stream")
    assert record is not None
    assert record.is_consumed
    assert record.parameters == {"city": "Paris"}

    assert len(ctx.messages) == 4
    assert isinstance(ctx.messages[2], ToolMessage)
    assert ctx.messages[2].content == "sunny"
    assert ctx.messages[3].text() == "Weather is sunny."
    assert state.llm_calls == 2
