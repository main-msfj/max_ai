"""Tests for Agent(..., store=...) run checkpointing.

When a store is configured the agent persists the RunContext at safe
points: after the task lands, before each reasoning iteration, and on
the terminal response (which covers pauses). A crashed process can then
load the checkpoint, apply pending decisions, and resume.
"""

from __future__ import annotations

import typing as t

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.models import ModelConfig
from max_ai.persistence import RunContextStore
from max_ai.termination import CancellationToken
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode, CoreToolParameters


class InMemoryStore(RunContextStore):
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.save_count = 0

    async def save(self, run_id: str, ctx: RunContext) -> None:
        self.save_count += 1
        self.data[run_id] = ctx.model_dump_json()

    async def load(self, run_id: str) -> RunContext | None:
        raw = self.data.get(run_id)
        return RunContext.model_validate_json(raw) if raw else None

    async def delete(self, run_id: str) -> None:
        self.data.pop(run_id, None)

    async def list(self) -> list[str]:
        return list(self.data)


class FailingStore(InMemoryStore):
    async def save(self, run_id: str, ctx: RunContext) -> None:
        raise RuntimeError("disk full")


class ScriptedClient(CoreChatCompletionClient):
    """Returns queued (content, tool_calls) responses in order."""

    def __init__(self, script: list[tuple[str, list[ToolCall] | None]]) -> None:
        super().__init__(model="scripted", config=ModelConfig())
        self._script = list(script)

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def format_messages(self, ctx, prompts):
        return list(ctx.messages)

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        content, tool_calls = self._script.pop(0)
        return ChatCompletionResult(
            message=AssistantMessage(
                source="scripted", content=content, tool_calls=tool_calls or []
            ),
            usage=Usage(llm_calls=1, attempts_to_call_api=1),
            model="scripted",
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def stream(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError


class GatedTool(CoreTool):
    def __init__(self) -> None:
        super().__init__(
            name="dangerous",
            description="Needs approval",
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )
        self.execute_called = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        return CoreToolParameters(is_tool_valid=True, msg_error=None)

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.execute_called = True
        return ToolResult.success_result(tool_request.id, "done dangerously")


def make_agent(client, store, toolset=None) -> Agent:
    return Agent(
        name="checkpointer",
        description="test agent",
        instructions="be concise",
        client=client,
        store=store,
        toolset=toolset,
    )


@pytest.mark.asyncio
async def test_run_checkpoints_to_store():
    store = InMemoryStore()
    agent = make_agent(ScriptedClient([("done", None)]), store)

    response = await agent.run("hello")

    assert response.finish_reason == "stop"
    assert store.save_count >= 2  # task landed + iteration + terminal
    saved = await store.load(response.context.run_id)
    assert saved is not None
    assert saved.messages[-1].text() == "done"


@pytest.mark.asyncio
async def test_pause_is_checkpointed_and_resumable_from_store():
    """Approval pause → checkpoint carries the pending record; a 'new
    process' loads it, applies the decision, and resumes to completion."""
    store = InMemoryStore()
    tool = GatedTool()
    tc = ToolCall(id="c_d", tool_name="dangerous", parameters={})
    client = ScriptedClient([("", [tc]), ("all done", None)])
    agent = make_agent(client, store, toolset=[tool])

    response = await agent.run("do the thing")
    assert response.finish_reason == "approval_needed"
    run_id = response.context.run_id

    # The pause survived to the store, including the pending approval and
    # the stashed loop metrics.
    loaded = await store.load(run_id)
    assert loaded is not None
    assert loaded.tool_state.waiting_for_approval
    assert "loop_metrics" in loaded.runtime_state.shared_state

    # New process: apply the decision on the loaded snapshot and resume.
    loaded.tool_state.apply_approval("c_d", approved=True)
    resumed = await agent.resume(loaded)

    assert resumed.finish_reason == "stop"
    assert tool.execute_called is True
    assert resumed.final_text == "all done"

    # Terminal state overwrote the checkpoint.
    final = await store.load(run_id)
    assert final is not None
    assert not final.tool_state.waiting_for_approval
    assert final.messages[-1].text() == "all done"


@pytest.mark.asyncio
async def test_checkpoint_failure_never_kills_the_run():
    agent = make_agent(ScriptedClient([("done", None)]), FailingStore())

    response = await agent.run("hello")

    assert response.finish_reason == "stop"
    assert response.final_text == "done"


@pytest.mark.asyncio
async def test_load_run_requires_store():
    agent = make_agent(ScriptedClient([("done", None)]), store=None)
    from max_ai.errors.agent import AgentError

    with pytest.raises(AgentError, match="no RunContextStore"):
        await agent.load_run("abc")


@pytest.mark.asyncio
async def test_no_store_means_no_persistence_side_effects():
    """Without a store the run behaves exactly as before."""
    agent = make_agent(ScriptedClient([("done", None)]), store=None)
    response = await agent.run("hello")
    assert response.finish_reason == "stop"
