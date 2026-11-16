"""MCP tasks: a call left running is kept in the run context and collected later."""

import time

import pytest
from mcp import MCPError
from mcp.types import CallToolResult, TextContent

from max_ai.capabilities.mcp import client_manager as cm
from max_ai.capabilities.mcp._model import HTTPServerConfig
from max_ai.capabilities.mcp.client_manager import MCPClientManager
from max_ai.capabilities.mcp.task import MCPTaskRunning, TaskState
from max_ai.capabilities.mcp.tool import TASK_RUNNING, MCPTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


def manager(monkeypatch, answers) -> tuple[MCPClientManager, list[str]]:
    """A manager whose server answers ``tasks/get`` from ``answers`` by task id."""
    asked: list[str] = []

    class Session:
        async def send_request(self, request, result_type):
            task_id = request.params.task_id
            asked.append(task_id)
            if task_id not in answers:
                raise MCPError(-32602, "unknown task")
            return TaskState.model_validate({"taskId": task_id, **answers[task_id]})

    class Client:
        session = Session()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *error):
            return False

    async def no_tools(self, config, client):
        return []

    monkeypatch.setattr(cm, "create_mcp_client", lambda config, extensions=None: Client())
    monkeypatch.setattr(MCPClientManager, "_discover_server_tools", no_tools)
    made = MCPClientManager()
    made.add_server(HTTPServerConfig(server_id="docs", url="http://localhost:1/mcp"))
    return made, asked


def kept(task_id: str, **more) -> dict:
    return {"task_id": task_id, "server_id": "docs", "tool_name": "docs_report", **more}


@pytest.mark.asyncio
async def test_a_call_left_running_is_kept_in_the_run_context():
    class Manager:
        max_timeout = 600

        async def call_tool(self, *args, **kwargs):
            raise MCPTaskRunning("task-1", ttl_ms=900_000)

    tool = MCPTool(
        "report", "Build the report.", {"type": "object"}, Manager(), "docs",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
    )
    ctx = RunContext(user_id="user-1", session_id="s1")
    record = ToolCallRecord(id="call-1", tool_name=tool.name, parameters={})
    result = await ToolDispatcher(ToolRegistry([tool])).dispatch(record, ctx)

    assert result.success and result.result == TASK_RUNNING
    task = ctx.mcp_tasks["task-1"]
    assert (task["server_id"], task["tool_name"], task["tool_call_id"]) == (
        "docs", "docs_report", "call-1",
    )
    assert task["expires_at"] > time.time()
    # It survives the run being saved and loaded.
    assert RunContext.model_validate_json(ctx.model_dump_json()).mcp_tasks == ctx.mcp_tasks


@pytest.mark.asyncio
async def test_collecting_returns_what_ended_and_keeps_what_still_runs(monkeypatch):
    done = CallToolResult(content=[TextContent(type="text", text="the report")])
    made, asked = manager(monkeypatch, {
        "running": {"status": "working"},
        "done": {"status": "completed", "result": done.model_dump(by_alias=True)},
        "failed": {"status": "failed", "error": {"message": "no data"}},
    })
    ctx = RunContext(user_id="user-1", session_id="s1")
    ctx.mcp_tasks = {
        "running": kept("running"), "done": kept("done"), "failed": kept("failed"),
        "gone": kept("gone"),
        "expired": kept("expired", expires_at=time.time() - 1),
    }
    try:
        ended = {task.task_id: result for task, result in await made.collect_tasks(ctx)}
    finally:
        await made.disconnect_all()

    assert list(ctx.mcp_tasks) == ["running"]
    assert ended["done"].content[0].text == "the report" and not ended["done"].is_error
    assert ended["failed"].is_error and "no data" in ended["failed"].content[0].text
    assert ended["gone"] is None and ended["expired"] is None
    assert "expired" not in asked  # past its time: the server is not even asked
