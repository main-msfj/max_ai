import pytest

from max_ai.base.tools import CoreTool, ToolContext
from max_ai.capabilities.reasoning.react.loop import ReactLoop, ReActLoopState
from max_ai.core.event_type import ReasoningCompleteEvent, ToolApprovalEvent
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.tool.dispatcher import ToolDispatcher
from max_ai.core.tool.registry import ToolRegistry
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolResult
from max_ai.types.tools import ToolApprovalMode


class _Config:
    tokenizer_base = "o200k_base"


class FakeClient:
    model = "fake"
    config = _Config()

    def __init__(self):
        self.calls = 0

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls += 1
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content="done"),
            usage=Usage(), model=self.model, finish_reason="stop",
        )


class ToolThenFinalClient(FakeClient):
    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatCompletionResult(
                message=AssistantMessage(source="fake", content="", tool_calls=[
                    ToolCall(id="call-1", tool_name="count", parameters={})
                ]), usage=Usage(), model=self.model, finish_reason="tool_calls",
            )
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content="final"),
            usage=Usage(), model=self.model, finish_reason="stop",
        )


class NeverCompletesStreamClient(FakeClient):
    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls += 1

        async def _stream():
            yield ChatCompletionChunk(content="partial", is_complete=False)

        return _stream()


class CountingTool(CoreTool):
    def __init__(self, mode=ToolApprovalMode.AUTO_APPROVED):
        super().__init__("count", "count", approval_mode=mode)
        self.calls = 0

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, record, context=None, cancellation_token=None):
        self.calls += 1
        return ToolResult.success_result(record.id, {"calls": self.calls})


def make_loop(client, tools=()):
    # host=True: tools run in-process, no EnvironmentManager needed for these tests.
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool, host=True)
    dispatcher = ToolDispatcher(registry)
    ctx = RunContext(user_id="u", session_id="c")
    tool_context = ToolContext(ctx.run_id, session_id=ctx.session_id, user_id=ctx.user_id)
    loop = ReactLoop(max_loop_iterations=5).bind(
        name="test", client=client, dispatcher=dispatcher, tool_context=tool_context,
    )
    return loop, ctx


async def drive(loop, ctx, prompts, loop_state=None, **kwargs):
    events = []
    state = loop_state if loop_state is not None else ReActLoopState()
    async for event in loop.execute_reasoning_loop(
        ctx=ctx, prompts=prompts, loop_state=state, **kwargs
    ):
        events.append(event)
    return events, state


@pytest.mark.asyncio
async def test_simple_response_completes_in_one_iteration(prompts):
    client = FakeClient()
    loop, ctx = make_loop(client)
    events, state = await drive(loop, ctx, prompts)
    assert state.finish_reason == "stop"
    assert state.iteration == 1
    assert isinstance(events[-1], ReasoningCompleteEvent)
    assert ctx.messages[-1].text() == "done"


@pytest.mark.asyncio
async def test_tool_call_then_final_response(prompts):
    client = ToolThenFinalClient()
    tool = CountingTool()
    loop, ctx = make_loop(client, [tool])
    events, state = await drive(loop, ctx, prompts)
    assert state.finish_reason == "stop"
    assert state.iteration == 2
    assert tool.calls == 1
    assert client.calls == 2
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "call-1" for m in ctx.messages
    )
    assert ctx.messages[-1].text() == "final"


@pytest.mark.asyncio
async def test_approval_pause_then_resume(prompts):
    client = ToolThenFinalClient()
    tool = CountingTool(mode=ToolApprovalMode.ASK_APPROVED)
    loop, ctx = make_loop(client, [tool])

    events, state = await drive(loop, ctx, prompts)
    assert state.finish_reason == "approval_needed"
    assert any(isinstance(e, ToolApprovalEvent) for e in events)
    assert tool.calls == 0
    pending = [r for r in ctx.tool_state.records.values() if not r.is_consumed]
    assert len(pending) == 1

    pending[0].approve("ok")
    events2, state2 = await drive(loop, ctx, prompts)
    assert state2.finish_reason == "stop"
    assert tool.calls == 1
    assert ctx.messages[-1].text() == "final"


@pytest.mark.asyncio
async def test_incomplete_stream_yields_no_llm_result_without_stale_reuse(prompts):
    # Regression test: loop_state.last_result used to persist across calls,
    # so a stream that never completes could get mistaken for a fresh result.
    client = NeverCompletesStreamClient()
    loop, ctx = make_loop(client)
    state = ReActLoopState()
    state.last_result = ChatCompletionResult(
        message=AssistantMessage(source="fake", content="", tool_calls=[
            ToolCall(id="stale-call", tool_name="count", parameters={})
        ]), usage=Usage(), model="fake", finish_reason="tool_calls",
    )

    events, _ = await drive(loop, ctx, prompts, loop_state=state, stream_tokens=True)

    assert state.finish_reason == "no_result"
    assert state.last_result is None
    assert ctx.tool_state.records == {}
    assert ctx.messages == []
    assert isinstance(events[-1], ReasoningCompleteEvent)
