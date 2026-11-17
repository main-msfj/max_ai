"""MCP: what a server says is not believed by default, a stdio command can
be restricted, and small failures do not lose data or stop a close."""

import asyncio
from types import SimpleNamespace

import httpx2
import pytest
from mcp.types import CallToolResult

from max_ai.capabilities.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    MCPTool,
    StdioMCPServerConfig,
)
from max_ai.capabilities.mcp.client_manager import MCPClientManager as _Manager

_unreachable = _Manager._unreachable
from max_ai.errors.mcp import MCPServerRegistrationError
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode

READ_ONLY = SimpleNamespace(
    name="search",
    annotations=SimpleNamespace(read_only_hint=True, destructive_hint=None, open_world_hint=None),
)


def http(**options) -> HTTPServerConfig:
    return HTTPServerConfig(server_id="docs", url="http://localhost:3000/mcp", **options)


def test_a_servers_own_hints_count_only_when_it_is_trusted():
    decide = MCPClientManager._approval_mode_for_tool
    assert decide(http(), READ_ONLY) == ToolApprovalMode.ASK_APPROVED
    assert decide(http(trusted=True), READ_ONLY) == ToolApprovalMode.AUTO_APPROVED
    # What the config says about a tool always wins.
    chosen = http(tool_approval_modes={"search": ToolApprovalMode.AUTO_APPROVED})
    assert decide(chosen, READ_ONLY) == ToolApprovalMode.AUTO_APPROVED


def test_a_stdio_command_can_be_limited_to_a_list():
    npx = StdioMCPServerConfig(server_id="fs", command="npx", args=["server"])
    MCPClientManager().add_server(npx)  # no list: any command, as before
    MCPClientManager(stdio_commands=["npx"]).add_server(npx)
    with pytest.raises(MCPServerRegistrationError, match="not allowed"):
        MCPClientManager(stdio_commands=["uvx"]).add_server(npx)
    with pytest.raises(MCPServerRegistrationError):
        MCPClientManager(stdio_commands=[]).add_server(npx)
    MCPClientManager(stdio_commands=[]).add_server(http())  # HTTP is not a command


def test_a_network_error_of_the_http_transport_is_worth_trying_again():
    assert _unreachable(httpx2.ConnectError("refused"))
    assert not _unreachable(ValueError("bad answer"))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"result": 5}, 5),
        ({"results": [1, 2]}, [1, 2]),
        ({"result": 5, "warnings": ["slow"]}, {"result": 5, "warnings": ["slow"]}),
        ({"total": 3}, {"total": 3}),
    ],
)
async def test_only_a_wrapped_plain_value_is_unwrapped(answer, expected):
    class Manager:
        async def call_tool(self, *args, **kwargs):
            return CallToolResult(content=[], structured_content=answer)

    tool = MCPTool("count", "Count the documents that match.", {"type": "object"}, Manager(), "docs")
    result = await tool.execute(ToolCallRecord(tool_name=tool.name, parameters={}))
    assert result.success and result.result == expected


async def test_closing_goes_on_when_a_connection_had_already_died():
    async def died():
        raise ConnectionError("the server went away")

    manager = MCPClientManager()
    for server_id in ("first", "second"):
        manager.add_server(HTTPServerConfig(server_id=server_id, url="http://localhost:1/mcp"))
        worker = asyncio.create_task(died())
        await asyncio.gather(worker, return_exceptions=True)
        manager._servers[server_id].worker = worker

    await manager.disconnect_all()
    assert all(server.worker is None for server in manager._servers.values())


# -------- images and other files -------------------------------------------------
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 100


async def image_call(tmp_path=None):
    """Dispatch an MCP tool that answers a sentence and a picture."""
    import base64

    from mcp.types import ImageContent, TextContent

    from max_ai.capabilities.sandbox.local import LocalSandbox
    from max_ai.core.tool import ToolDispatcher, ToolRegistry
    from max_ai.types.run_context import RunContext

    class Manager:
        async def call_tool(self, *args, **kwargs):
            picture = base64.b64encode(PNG).decode()
            return CallToolResult(content=[
                TextContent(type="text", text="Chart for Q3 sales."),
                ImageContent(type="image", data=picture, mime_type="image/png"),
            ])

    tool = MCPTool(
        "chart", "Draw the sales chart of a quarter.", {"type": "object"}, Manager(), "docs",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
    )
    sandbox = None
    if tmp_path is not None:
        sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
        await sandbox.prepare()
    record = ToolCallRecord(id="call/abc", tool_name=tool.name, parameters={})
    ctx = RunContext(user_id="user-1", session_id="s1")
    result = await ToolDispatcher(ToolRegistry([tool]), sandbox).dispatch(record, ctx)
    return record, result


async def test_an_image_is_kept_in_the_sandbox_and_the_model_gets_its_path(tmp_path):
    record, result = await image_call(tmp_path)
    saved = tmp_path / ".agent/tmp/s1/tool_results/call_abc-1.png"
    assert saved.read_bytes() == PNG
    assert result.result == (
        f"Chart for Q3 sales.\n\nThe tool returned 1 file(s):\n- image/png, 1 KB: {saved}"
    )
    # Nothing of the picture stays in what is saved with the session.
    assert "files" not in record.model_dump_json() and result.files == []


async def test_without_a_sandbox_the_model_is_told_what_the_file_was():
    _, result = await image_call()
    assert result.result.endswith("- image/png, 1 KB: could not be kept, so it cannot be shown")
