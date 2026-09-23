import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.reasoning.react.loop import ReactLoop, ReActLoopState
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.tool.dispatcher import ToolDispatcher
from max_ai.core.tool.registry import ToolRegistry
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class _Config:
    tokenizer_base = "o200k_base"


class RaisingClient:
    model = "fake"
    config = _Config()

    async def run(self, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        raise RuntimeError("boom")


def make_loop(client):
    dispatcher = ToolDispatcher(ToolRegistry())
    ctx = RunContext(user_id="u", session_id="c")
    tool_context = ToolContext(ctx.run_id, session_id=ctx.session_id, user_id=ctx.user_id)
    loop = ReactLoop().bind(
        name="test", client=client, dispatcher=dispatcher, tool_context=tool_context,
    )
    return loop, ctx


def stale_result() -> ChatCompletionResult:
    return ChatCompletionResult(
        message=AssistantMessage(source="fake", content="", tool_calls=[
            ToolCall(id="stale", tool_name="x", parameters={})
        ]), usage=Usage(), model="fake", finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_call_llm_resets_stale_last_result_before_failing(prompts):
    loop, ctx = make_loop(RaisingClient())
    state = ReActLoopState()
    state.last_result = stale_result()

    with pytest.raises(RuntimeError):
        async for _ in loop._call_llm(ctx=ctx, prompts=prompts, loop_state=state):
            pass

    assert state.last_result is None


@pytest.mark.asyncio
async def test_call_llm_stream_resets_stale_last_result_before_failing(prompts):
    loop, ctx = make_loop(RaisingClient())
    state = ReActLoopState()
    state.last_result = stale_result()

    with pytest.raises(RuntimeError):
        async for _ in loop._call_llm_stream(ctx=ctx, prompts=prompts, loop_state=state):
            pass

    assert state.last_result is None


@pytest.mark.asyncio
async def test_loop_rejects_wrong_loop_state_type(prompts):
    from max_ai.base.reasoning import BaseLoopState

    loop, ctx = make_loop(RaisingClient())
    with pytest.raises(TypeError):
        async for _ in loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=BaseLoopState(),
        ):
            pass
