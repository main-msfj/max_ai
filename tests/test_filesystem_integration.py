from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

from max_ai.base.agent import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.tool_executor import ToolExecutor
from max_ai.core.event_type import ToolApprovalEvent
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.models import ModelConfig
from max_ai.tools import FileSystemTools
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.workspace_copy.system import LocalWorkSpace


async def _collect(generator) -> list[t.Any]:
    return [item async for item in generator]


@pytest.mark.asyncio
async def test_real_file_tool_pauses_without_writing_then_writes_after_approval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Workspace"
    tools = FileSystemTools().tools
    executor = ToolExecutor(
        tools=tools,
        agent_name="filesystem-integration",
        runtime_deps={"filesystem_root": str(root)},
    )
    context = RunContext(user_id="tenant-1", session_id="conversation-1")
    record = ToolCallRecord(
        tool_name="write_file",
        parameters={"path": "notes/summary.txt", "content": "approved text"},
    )
    context.tool_state.add(record)
    target = root / "tenant-1" / "conversation-1" / "notes" / "summary.txt"

    paused_events = await _collect(executor.execute_tool_call(context, [record]))

    assert any(isinstance(event, ToolApprovalEvent) for event in paused_events)
    assert record.is_pending_approval
    assert not target.exists()

    context.tool_state.apply_approval(record.id, approved=True, reason="approved")
    resumed_events = await _collect(
        executor.execute_tool_call(context, context.tool_state.actionable_calls)
    )

    assert not any(isinstance(event, ToolApprovalEvent) for event in resumed_events)
    assert target.read_text(encoding="utf-8") == "approved text"
    assert record.is_consumed
    assert record.result is not None and record.result.success is True
    assert record.result.result["path"] == "conversation-1/notes/summary.txt"


class FilesystemClient(CoreChatCompletionClient):
    def __init__(self) -> None:
        super().__init__(model="filesystem-test", config=ModelConfig())
        self.responses = [
            ChatCompletionResult(
                message=AssistantMessage(
                    source="filesystem-test",
                    tool_calls=[
                        ToolCall(
                            id="write-call",
                            tool_name="write_file",
                            parameters={"path": "report.txt", "content": "from agent"},
                        )
                    ],
                ),
                usage=Usage(llm_calls=1, attempts_to_call_api=1),
                model="filesystem-test",
                finish_reason="tool_calls",
            ),
            ChatCompletionResult(
                message=AssistantMessage(source="filesystem-test", content="saved"),
                usage=Usage(llm_calls=1, attempts_to_call_api=1),
                model="filesystem-test",
                finish_reason="stop",
            ),
        ]

    def normalize_usage_stats(self, usage: t.Any) -> Usage:
        return Usage()

    def format_messages(self, ctx, prompts):
        return list(ctx.messages)

    def build_api_messages(self, messages):
        return []

    def build_tool_schema(self, tools):
        return []

    async def complete(self, messages, tools, output_format, **kwargs):
        return self.responses.pop(0)

    async def stream(self, messages, tools, output_format, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_agent_uses_workspace_override_context_ids_and_run_id_session_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "custom-workspace"
    agent = Agent(
        name="filesystem-agent",
        description="filesystem integration test",
        instructions="Use the file tools.",
        client=FilesystemClient(),
        workspace=LocalWorkSpace(root=root),
    )
    context = RunContext(
        user_id="trusted-user",
        session_id=None,
        run_id="stable-run-id",
    )
    tool = next(
        tool for tool in agent.registries._all_tools_sync() if tool.name == "write_file"
    )
    exposed_parameters = set(tool.parameters["properties"])
    assert {"user_id", "session_id", "filesystem_root"}.isdisjoint(exposed_parameters)

    paused = await agent.run("Save a report.", run_context=context)

    assert paused.finish_reason == "approval_needed"
    assert paused.context is not None
    assert paused.context.session_id == "stable-run-id"
    target = root / "trusted-user" / "stable-run-id" / "report.txt"
    assert not target.exists()

    pending = paused.pending_approvals[0]
    paused.context.tool_state.apply_approval(
        pending.id,
        approved=True,
        reason="approved in integration test",
    )
    completed = await agent.resume(run_context=paused.context)

    assert completed.finish_reason == "stop"
    assert target.read_text(encoding="utf-8") == "from agent"
    assert (root / "trusted-user" / "stable-run-id").is_dir()
    assert not (root / "untrusted-user").exists()
