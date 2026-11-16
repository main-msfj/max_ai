"""A long MCP call: progress notices keep it alive and reach the caller."""

import asyncio

import pytest

from max_ai.capabilities.mcp import client_manager as cm
from max_ai.capabilities.mcp._model import HTTPServerConfig
from max_ai.capabilities.mcp.client_manager import MCPClientManager
from max_ai.errors.mcp import MCPToolError


def manager(monkeypatch, call_tool, **limits) -> MCPClientManager:
    """A manager whose only server answers with ``call_tool``."""

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *error):
            return False

        async def list_tools(self, cursor=None):
            raise NotImplementedError

    Client.call_tool = call_tool

    async def no_tools(self, config, client):
        return []

    monkeypatch.setattr(cm, "create_mcp_client", lambda config, extensions=None: Client())
    monkeypatch.setattr(MCPClientManager, "_discover_server_tools", no_tools)
    made = MCPClientManager(**limits)
    made.add_server(HTTPServerConfig(server_id="slow", url="http://localhost:1/mcp"))
    return made


@pytest.mark.asyncio
async def test_progress_keeps_a_call_alive_and_reaches_the_caller(monkeypatch):
    async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
        for step in range(1, 5):  # 0.4s of work, a notice every 0.1s
            await asyncio.sleep(0.1)
            await progress_callback(step, 4, f"step {step}")
        return "done"

    made = manager(monkeypatch, call_tool, idle_timeout=0.25, max_timeout=5)
    seen = []
    try:
        result = await made.call_tool(
            "slow", "work", {}, on_progress=lambda *notice: seen.append(notice)
        )
    finally:
        await made.disconnect_all()

    assert result == "done"
    assert seen == [(1, 4, "step 1"), (2, 4, "step 2"), (3, 4, "step 3"), (4, 4, "step 4")]


@pytest.mark.asyncio
async def test_a_silent_call_is_cut_at_the_idle_limit(monkeypatch):
    async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
        await asyncio.sleep(30)

    made = manager(monkeypatch, call_tool, idle_timeout=0.2, max_timeout=5)
    try:
        with pytest.raises(MCPToolError, match="stopped answering"):
            await made.call_tool("slow", "work", {})
    finally:
        await made.disconnect_all()


@pytest.mark.asyncio
async def test_progress_does_not_pass_the_total_limit(monkeypatch):
    async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None):
        while True:
            await asyncio.sleep(0.05)
            await progress_callback(1, None, None)

    made = manager(monkeypatch, call_tool, idle_timeout=0.2, max_timeout=0.5)
    try:
        with pytest.raises(MCPToolError, match="stopped answering"):
            await made.call_tool("slow", "work", {})
    finally:
        await made.disconnect_all()


@pytest.mark.asyncio
async def test_the_tool_turns_progress_into_an_event_and_sets_its_own_limit():
    from mcp.types import CallToolResult, TextContent

    from max_ai.capabilities.mcp.tool import MCPTool
    from max_ai.core.event_type import MCPProgressEvent
    from max_ai.core.tool import ToolDispatcher, ToolRegistry
    from max_ai.types.run_context import RunContext
    from max_ai.types.tool_call import ToolCallRecord
    from max_ai.types.tools import ToolApprovalMode

    class Manager:
        max_timeout = 600

        async def call_tool(self, *args, on_progress=None, **kwargs):
            on_progress(30, 100, "downloading")
            return CallToolResult(content=[TextContent(type="text", text="done")])

    tool = MCPTool(
        "fetch", "Download the report.", {"type": "object"}, Manager(), "docs",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
    )
    assert tool.timeout == 600

    seen = []
    record = ToolCallRecord(id="call-1", tool_name=tool.name, parameters={})
    ctx = RunContext(user_id="user-1", session_id="s1")
    result = await ToolDispatcher(ToolRegistry([tool])).dispatch(record, ctx, seen.append)

    progress = [event for event in seen if isinstance(event, MCPProgressEvent)]
    assert result.success and len(progress) == 1
    assert (progress[0].progress, progress[0].total, progress[0].message) == (30, 100, "downloading")
    assert progress[0].tool_call_id == "call-1"
