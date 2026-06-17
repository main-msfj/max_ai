"""Tests for Human-in-the-Loop (UserInputTool + resume) in both ReAct loops."""

from __future__ import annotations

import asyncio
import pytest
import typing as t

from max_ai.reasoning.react_simple import (
    ReActLoop as SimpleReActLoop,
    ReActLoopState as SimpleLoopState,
)
from max_ai.reasoning.react_planning import (
    ReActLoopPlanning as PlanningReActLoop,
    ReActLoopPlanningState as PlanningLoopState,
)
from max_ai.reasoning.plan import AgentPlan
from max_ai.core.messages import AssistantMessage, ToolMessage
from max_ai.core.event_type import ReasoningCompleteEvent, UserInputRequestEvent
from max_ai.core.models import ModelConfig
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.tools.structure_human_in_loop import UserInputTool
from max_ai.base.reasoning import BaseLoopState


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list):
        self.model = "fake"
        self.config = ModelConfig()
        self._results = list(results)
        self._call_count = 0

    async def run(
        self, ctx, prompts, tools=None, output_format=None, stream=False, **kw
    ):
        self._call_count += 1
        return self._results.pop(0)

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


# class FakeToolExecutor:
#     """Executor that can optionally run a real UserInputTool."""

#     def __init__(self, tool: UserInputTool | None = None):
#         self._tool = tool
#         self.tools: dict = {}

#     async def execute_tool_call(self, ctx, records, cancellation_token=None):
#         for record in records:
#             if self._tool and record.tool_name == UserInputTool.TOOL_NAME:
#                 result = await self._tool.execute(record)
#                 yield ToolMessage(
#                     source="tool",
#                     tool_call_id=record.id,
#                     tool_name=record.tool_name,
#                     success=result.success,
#                     content=result.result or "",
#                 )
#             else:
#                 yield ToolMessage(
#                     source="tool",
#                     tool_call_id=record.id,
#                     tool_name=record.tool_name,
#                     success=True,
#                     content="ok",
#                 )

class FakeToolExecutor:
    """Executor that runs a hand-passed UserInputTool OR whatever the loop
    auto-registered into self.tools."""

    def __init__(self, tool: UserInputTool | None = None):
        self._tool = tool
        self.tools: dict = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            if self._tool and record.tool_name == UserInputTool.TOOL_NAME:
                result = await self._tool.execute(record)
            elif record.tool_name in self.tools:
                result = await self.tools[record.tool_name].execute(record)
            else:
                yield ToolMessage(
                    source="tool",
                    tool_call_id=record.id,
                    tool_name=record.tool_name,
                    success=True,
                    content="ok",
                )
                continue

            yield ToolMessage(
                source="tool",
                tool_call_id=record.id,
                tool_name=record.tool_name,
                success=result.success,
                content=result.result or "",
            )


class FakeMiddlewareChain:
    async def execute(self, action, ctx, data, func, metadata=None):
        yield await func(ctx)

    async def execute_stream(self, action, ctx, data, stream_func, metadata=None):
        async for chunk in stream_func(ctx):
            yield chunk


# -------- HELPERS -------------------------------------------------------------
def make_result(content="done", tool_calls=None):
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


def make_tool_call_result(tool_name: str, parameters: dict, call_id: str = "call_001"):
    from max_ai.core.messages import ToolCall

    return make_result(
        content="",
        tool_calls=[ToolCall(id=call_id, tool_name=tool_name, parameters=parameters)],
    )


def make_simple_loop(client, tool: UserInputTool | None = None):
    loop = SimpleReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(tool=tool),
        middleware_chain=FakeMiddlewareChain(),
    )
    return loop


def make_planning_loop(client, tool: UserInputTool | None = None):
    loop = PlanningReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(tool=tool),
        middleware_chain=FakeMiddlewareChain(),
    )
    return loop


@pytest.fixture
def ctx():
    return RunContext()


@pytest.fixture
def prompts():
    return PromptCtx.model_construct(stack=None, variables={}, rendered_layers={})


async def collect_until_input(gen) -> tuple[list, t.AsyncGenerator]:
    """Collect events until UserInputRequestEvent, return (events_so_far, generator)."""
    events = []
    async for ev in gen:
        events.append(ev)
        if isinstance(ev, UserInputRequestEvent):
            return events, gen
    return events, gen


# -------- UserInputTool unit tests --------------------------------------------
@pytest.mark.asyncio
async def test_user_input_tool_sets_future_and_finish_reason():
    """execute() sets pending_user_input Future and finish_reason=input_needed."""
    loop_state = BaseLoopState()
    tool = UserInputTool(loop_state=loop_state)

    record = ToolCallRecord(
        id="call_1",
        tool_name=UserInputTool.TOOL_NAME,
        parameters={"question": "What format?", "options": ["JSON", "CSV"]},
        session_id="s1",
    )

    async def _execute_and_resolve():
        # Start execute — it will block on await future
        task = asyncio.create_task(tool.execute(record))
        # Wait until the future is set on loop_state
        for _ in range(50):
            if loop_state.pending_user_input is not None:
                break
            await asyncio.sleep(0.01)

        assert loop_state.finish_reason == "input_needed"
        assert loop_state.pending_user_input_question == "What format?"
        assert loop_state.pending_user_input_options == ["JSON", "CSV"]

        # Resolve
        loop_state.pending_user_input.set_result("JSON")
        return await task

    result = await _execute_and_resolve()
    assert result.success is True
    assert result.result == "JSON"


@pytest.mark.asyncio
async def test_user_input_tool_no_options():
    """options is optional — free-text question works without options."""
    loop_state = BaseLoopState()
    tool = UserInputTool(loop_state=loop_state)

    record = ToolCallRecord(
        id="call_1",
        tool_name=UserInputTool.TOOL_NAME,
        parameters={"question": "What is your name?"},
        session_id="s1",
    )

    async def _run():
        task = asyncio.create_task(tool.execute(record))
        for _ in range(50):
            if loop_state.pending_user_input is not None:
                break
            await asyncio.sleep(0.01)
        assert loop_state.pending_user_input_options is None
        loop_state.pending_user_input.set_result("Marvin")
        return await task

    result = await _run()
    assert result.result == "Marvin"


@pytest.mark.asyncio
async def test_user_input_tool_parameters_schema():
    """Tool exposes correct JSON schema with question required, options optional."""
    loop_state = BaseLoopState()
    tool = UserInputTool(loop_state=loop_state)

    schema = tool.parameters
    assert schema["type"] == "object"
    assert "question" in schema["properties"]
    assert "options" in schema["properties"]
    assert "question" in schema["required"]
    assert "options" not in schema.get("required", [])


# -------- resume() tests ------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_resolves_future():
    """loop.resume(answer) resolves the pending Future."""
    loop = make_simple_loop(FakeChatClient([]))
    loop_state = SimpleLoopState()
    loop._set_loop_state(loop_state)

    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    loop_state.pending_user_input = future

    loop.resume("hello")
    assert future.done()
    assert future.result() == "hello"


@pytest.mark.asyncio
async def test_resume_raises_when_no_active_loop():
    """resume() raises RuntimeError when no loop is active."""
    loop = make_simple_loop(FakeChatClient([]))
    with pytest.raises(RuntimeError, match="No active reasoning loop"):
        loop.resume("answer")


@pytest.mark.asyncio
async def test_resume_raises_when_no_pending_input():
    """resume() raises RuntimeError when loop is active but no pending input."""
    loop = make_simple_loop(FakeChatClient([]))
    loop_state = SimpleLoopState()
    loop._set_loop_state(loop_state)
    # pending_user_input is None
    with pytest.raises(RuntimeError, match="No pending user input"):
        loop.resume("answer")


# -------- Integration: simple ReActLoop ---------------------------------------
@pytest.mark.asyncio
async def test_simple_loop_emits_user_input_event(ctx, prompts):
    """LLM calls request_user_input → loop yields UserInputRequestEvent + complete."""
    loop_state = SimpleLoopState()
    tool = UserInputTool(loop_state=loop_state)

    client = FakeChatClient(
        results=[
            make_tool_call_result(
                tool_name=UserInputTool.TOOL_NAME,
                parameters={"question": "Which format?", "options": ["JSON", "CSV"]},
            ),
        ]
    )
    loop = make_simple_loop(client, tool=tool)

    # Collect events — the loop will pause waiting for resume()
    events_task = asyncio.create_task(
        _collect_events(
            loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state)
        )
    )

    # Wait until UserInputRequestEvent is emitted, then resume
    for _ in range(100):
        await asyncio.sleep(0.01)
        if loop_state.pending_user_input is not None:
            break

    assert loop_state.finish_reason == "input_needed"
    loop.resume("JSON")

    events = await events_task

    input_events = [e for e in events if isinstance(e, UserInputRequestEvent)]
    assert len(input_events) == 1
    assert input_events[0].question == "Which format?"
    assert input_events[0].options == ["JSON", "CSV"]

    complete_events = [e for e in events if isinstance(e, ReasoningCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].finish_reason == "input_needed"


@pytest.mark.asyncio
async def test_simple_loop_finish_reason_input_needed(ctx, prompts):
    """finish_reason is 'input_needed' when loop pauses for user input."""
    loop_state = SimpleLoopState()
    tool = UserInputTool(loop_state=loop_state)

    client = FakeChatClient(
        results=[
            make_tool_call_result(
                tool_name=UserInputTool.TOOL_NAME,
                parameters={"question": "Continue?"},
            ),
        ]
    )
    loop = make_simple_loop(client, tool=tool)

    events_task = asyncio.create_task(
        _collect_events(
            loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state)
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if loop_state.pending_user_input is not None:
            break

    loop.resume("yes")
    await events_task

    assert loop_state.finish_reason == "input_needed"


# -------- Integration: planning ReActLoop -------------------------------------
@pytest.mark.asyncio
async def test_planning_loop_emits_user_input_event(ctx, prompts):
    """Planning loop also yields UserInputRequestEvent correctly."""
    # Inert plan so the loop skips its planning step (this test is about the
    # user-input event, not planning).
    ctx.plan = AgentPlan(steps=[], rationale="no steps")
    loop_state = PlanningLoopState()
    tool = UserInputTool(loop_state=loop_state)

    client = FakeChatClient(
        results=[
            make_tool_call_result(
                tool_name=UserInputTool.TOOL_NAME,
                parameters={"question": "Which tone?", "options": ["formal", "casual"]},
            ),
        ]
    )
    loop = make_planning_loop(client, tool=tool)

    events_task = asyncio.create_task(
        _collect_events(
            loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state)
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if loop_state.pending_user_input is not None:
            break

    loop.resume("formal")
    events = await events_task

    input_events = [e for e in events if isinstance(e, UserInputRequestEvent)]
    assert len(input_events) == 1
    assert input_events[0].question == "Which tone?"
    assert input_events[0].options == ["formal", "casual"]


# -------- Helpers -------------------------------------------------------------
async def _collect_events(gen) -> list:
    return [ev async for ev in gen]


async def test_loop_auto_registers_human_input_tool(ctx, prompts):
    executor = FakeToolExecutor()
    loop = SimpleReActLoop(max_loop_iterations=5)
    loop.bind(
        name="t",
        client=FakeChatClient([make_result("hi")]),
        tool_executor=executor,
        middleware_chain=FakeMiddlewareChain(),
    )
    loop_state = SimpleLoopState()
    [
        ev
        async for ev in loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=loop_state
        )
    ]
    assert UserInputTool.TOOL_NAME in executor.tools


async def test_disable_human_input_skips_registration(ctx, prompts):
    executor = FakeToolExecutor()
    loop = SimpleReActLoop(max_loop_iterations=5, enable_human_input=False)
    loop.bind(
        name="t",
        client=FakeChatClient([make_result("hi")]),
        tool_executor=executor,
        middleware_chain=FakeMiddlewareChain(),
    )
    loop_state = SimpleLoopState()
    [
        ev
        async for ev in loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=loop_state
        )
    ]
    assert UserInputTool.TOOL_NAME not in executor.tools


async def test_self_directed_registries_both_tools(ctx, prompts):
    from max_ai.reasoning.react_self_directed import ReActLoopSelfDirected
    from max_ai.tools.update_plan import UpdatePlanTool

    executor = FakeToolExecutor()
    loop = ReActLoopSelfDirected(max_loop_iterations=5)
    loop.bind(
        name="t",
        client=FakeChatClient([make_result("hi")]),
        tool_executor=executor,
        middleware_chain=FakeMiddlewareChain(),
    )
    loop_state = SimpleLoopState()
    [
        ev
        async for ev in loop.execute_reasoning_loop(
            ctx=ctx, prompts=prompts, loop_state=loop_state
        )
    ]
    assert UserInputTool.TOOL_NAME in executor.tools
    assert UpdatePlanTool.TOOL_NAME in executor.tools


async def test_simple_loop_runs_auto_registered_tool_end_to_end(ctx, prompts):
    executor = FakeToolExecutor() 
    loop = SimpleReActLoop(max_loop_iterations=5)
    loop.bind(
        name="t",
        client=FakeChatClient(
            [
                make_tool_call_result(
                    UserInputTool.TOOL_NAME,
                    {"question": "Which format?", "options": ["JSON", "CSV"]},
                )
            ]
        ),
        tool_executor=executor,
        middleware_chain=FakeMiddlewareChain(),
    )
    loop_state = SimpleLoopState()

    events_task = asyncio.create_task(
        _collect_events(
            loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state)
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if loop_state.pending_user_input is not None:
            break

    loop.resume("JSON")
    events = await events_task

    input_events = [e for e in events if isinstance(e, UserInputRequestEvent)]
    assert len(input_events) == 1
    assert input_events[0].question == "Which format?"
