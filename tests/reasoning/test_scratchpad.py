"""Tests for ScratchpadTool and ScratchpadUpdateEvent in both ReAct loops."""

from __future__ import annotations

import pytest

from max_ai.reasoning.react_simple import ReActLoop as SimpleReActLoop, ReActLoopState as SimpleLoopState
from max_ai.reasoning.react_planning import ReActLoop as PlanningReActLoop, ReActLoopState as PlanningLoopState
from max_ai.core.messages import AssistantMessage, ToolMessage
from max_ai.core.event_type import ReasoningCompleteEvent, ScratchpadUpdateEvent
from max_ai.core.models import ModelConfig
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.tools.scratchpad import ScratchpadTool
from max_ai.base.scratchpad import Scratchpad, TodoItem
from max_ai.base.reasoning import BaseLoopState


# -------- FAKES ---------------------------------------------------------------
class FakeChatClient:
    def __init__(self, results: list):
        self.model = "fake"
        self.config = ModelConfig()
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
    def __init__(self, tool: ScratchpadTool | None = None):
        self._tool = tool
        self.tools: dict = {}

    async def execute_tool_call(self, ctx, records, cancellation_token=None):
        for record in records:
            if self._tool and record.tool_name == ScratchpadTool.TOOL_NAME:
                result = await self._tool.execute(record)
                yield ToolMessage(
                    source="tool",
                    tool_call_id=record.id,
                    tool_name=record.tool_name,
                    success=result.success,
                    content=result.result or "",
                )
            else:
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


def make_todo_call(id: str, description: str, status: str, call_id: str = "call_001"):
    from max_ai.core.messages import ToolCall
    return make_result(
        content="",
        tool_calls=[ToolCall(
            id=call_id,
            tool_name=ScratchpadTool.TOOL_NAME,
            parameters={"id": id, "description": description, "status": status},
        )],
    )


def make_simple_loop(client, tool: ScratchpadTool | None = None):
    loop = SimpleReActLoop(max_loop_iterations=5)
    loop.bind(
        name="test_agent",
        client=client,
        tool_executor=FakeToolExecutor(tool=tool),
        middleware_chain=FakeMiddlewareChain(),
    )
    return loop


def make_planning_loop(client, tool: ScratchpadTool | None = None):
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


async def collect(gen) -> list:
    return [ev async for ev in gen]


# -------- Scratchpad model tests ----------------------------------------------
def test_scratchpad_upsert_new_item():
    """upsert adds a new item when id doesn't exist."""
    pad = Scratchpad()
    pad.upsert(id="task_1", description="Search web", status="pending")
    assert len(pad.items) == 1
    assert pad.items[0].id == "task_1"
    assert pad.items[0].status == "pending"


def test_scratchpad_upsert_updates_existing():
    """upsert updates status of existing item — no duplicate."""
    pad = Scratchpad()
    pad.upsert(id="task_1", description="Search web", status="pending")
    pad.upsert(id="task_1", description="Search web", status="done")
    assert len(pad.items) == 1
    assert pad.items[0].status == "done"


def test_scratchpad_upsert_multiple_items():
    """Multiple distinct ids create multiple items."""
    pad = Scratchpad()
    pad.upsert(id="task_1", description="Step 1", status="done")
    pad.upsert(id="task_2", description="Step 2", status="in_progress")
    pad.upsert(id="task_3", description="Step 3", status="pending")
    assert len(pad.items) == 3


def test_scratchpad_upsert_updates_description():
    """upsert can update description of existing item."""
    pad = Scratchpad()
    pad.upsert(id="task_1", description="Old description", status="pending")
    pad.upsert(id="task_1", description="New description", status="in_progress")
    assert pad.items[0].description == "New description"


# -------- ScratchpadTool unit tests -------------------------------------------
@pytest.mark.asyncio
async def test_scratchpad_tool_sets_updated_flag():
    """execute() sets scratchpad_updated=True on loop_state."""
    loop_state = BaseLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    record = ToolCallRecord(
        id="call_1",
        tool_name=ScratchpadTool.TOOL_NAME,
        parameters={"id": "step_1", "description": "Search competitors", "status": "in_progress"},
        session_id="s1",
    )
    result = await tool.execute(record)

    assert result.success is True
    assert loop_state.scratchpad_updated is True
    assert len(loop_state.scratchpad.items) == 1
    assert loop_state.scratchpad.items[0].id == "step_1"
    assert loop_state.scratchpad.items[0].status == "in_progress"


@pytest.mark.asyncio
async def test_scratchpad_tool_does_not_pause():
    """ScratchpadTool returns immediately — no Future, no blocking."""
    loop_state = BaseLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    record = ToolCallRecord(
        id="call_1",
        tool_name=ScratchpadTool.TOOL_NAME,
        parameters={"id": "t1", "description": "Task", "status": "done"},
        session_id="s1",
    )
    result = await tool.execute(record)

    assert result.success is True
    assert loop_state.pending_user_input is None  # no Future created


@pytest.mark.asyncio
async def test_scratchpad_tool_result_message():
    """execute() returns informative result string."""
    loop_state = BaseLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    record = ToolCallRecord(
        id="call_1",
        tool_name=ScratchpadTool.TOOL_NAME,
        parameters={"id": "my_task", "description": "Do something", "status": "done"},
        session_id="s1",
    )
    result = await tool.execute(record)
    assert "my_task" in result.result
    assert "done" in result.result


# -------- Integration: simple ReActLoop ---------------------------------------
@pytest.mark.asyncio
async def test_simple_loop_emits_scratchpad_event(ctx, prompts):
    """LLM calls update_todo → loop yields ScratchpadUpdateEvent and continues."""
    loop_state = SimpleLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    client = FakeChatClient(results=[
        make_todo_call(id="step_1", description="Gather data", status="in_progress"),
        make_result(content="All done"),
    ])
    loop = make_simple_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state))

    scratchpad_events = [e for e in events if isinstance(e, ScratchpadUpdateEvent)]
    assert len(scratchpad_events) == 1
    assert scratchpad_events[0].scratchpad.items[0].id == "step_1"
    assert scratchpad_events[0].scratchpad.items[0].status == "in_progress"


@pytest.mark.asyncio
async def test_simple_loop_continues_after_scratchpad(ctx, prompts):
    """Loop does NOT stop after ScratchpadUpdateEvent — finish_reason is 'stop'."""
    loop_state = SimpleLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    client = FakeChatClient(results=[
        make_todo_call(id="step_1", description="Gather data", status="in_progress"),
        make_result(content="Final answer"),
    ])
    loop = make_simple_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state))

    complete = next(e for e in events if isinstance(e, ReasoningCompleteEvent))
    assert complete.finish_reason == "stop"


@pytest.mark.asyncio
async def test_simple_loop_multiple_scratchpad_updates(ctx, prompts):
    """Multiple update_todo calls → multiple ScratchpadUpdateEvents."""
    loop_state = SimpleLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    client = FakeChatClient(results=[
        make_todo_call(id="step_1", description="Step 1", status="in_progress", call_id="c1"),
        make_todo_call(id="step_1", description="Step 1", status="done", call_id="c2"),
        make_result(content="Finished"),
    ])
    loop = make_simple_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state))

    scratchpad_events = [e for e in events if isinstance(e, ScratchpadUpdateEvent)]
    assert len(scratchpad_events) == 2
    assert scratchpad_events[0].scratchpad.items[0].status == "in_progress"
    assert scratchpad_events[1].scratchpad.items[0].status == "done"


# -------- Integration: planning ReActLoop ------------------------------------
@pytest.mark.asyncio
async def test_planning_loop_emits_scratchpad_event(ctx, prompts):
    """Planning loop also emits ScratchpadUpdateEvent correctly."""
    loop_state = PlanningLoopState()
    tool = ScratchpadTool(loop_state=loop_state)

    client = FakeChatClient(results=[
        make_todo_call(id="research", description="Research topic", status="done"),
        make_result(content="Report ready"),
    ])
    loop = make_planning_loop(client, tool=tool)

    events = await collect(loop.execute_reasoning_loop(ctx=ctx, prompts=prompts, loop_state=loop_state))

    scratchpad_events = [e for e in events if isinstance(e, ScratchpadUpdateEvent)]
    assert len(scratchpad_events) == 1
    assert scratchpad_events[0].scratchpad.items[0].id == "research"
    assert scratchpad_events[0].scratchpad.items[0].status == "done"
