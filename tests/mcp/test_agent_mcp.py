from __future__ import annotations

import pytest
from mcp import Client
from mcp.server import MCPServer

from max_ai.agents import Agent
from max_ai.capabilities.mcp import MCPClientManager, StdioMCPServerConfig
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode
from tests.middleware.fakes import FakeClient, say


@pytest.mark.asyncio
async def test_manager_uses_v2_client_for_multiple_servers(monkeypatch):
    from max_ai.capabilities.mcp import client_manager as module

    servers = {}
    for server_id in ("one", "two"):
        server = MCPServer(server_id)

        @server.tool(name="echo")
        async def echo(value: str) -> str:
            return value

        servers[server_id] = server

    monkeypatch.setattr(module, "create_mcp_client", lambda config, extensions=None: Client(servers[config.server_id]))
    manager = MCPClientManager()
    for server_id in servers:
        manager.add_server(StdioMCPServerConfig(
            server_id=server_id, command="unused",
            tool_approval_modes={"echo": ToolApprovalMode.AUTO_APPROVED},
        ))

    await manager.connect_all()
    try:
        tools = manager.get_tools()
        assert [tool.name for tool in tools] == ["one_echo", "two_echo"]
        for tool in tools:
            result = await tool.execute(ToolCallRecord(
                tool_name=tool.name, parameters={"value": tool.server_id},
            ))
            assert result.success is True
            assert result.result == tool.server_id
    finally:
        await manager.disconnect_all()
    assert manager.get_tools() == []


@pytest.mark.asyncio
async def test_agent_shares_one_mcp_connection_until_close(monkeypatch):
    from max_ai.capabilities.mcp import client_manager as module

    server = MCPServer("docs")
    searched: list[str] = []

    @server.tool()
    async def search(q: str) -> str:
        searched.append(q)
        return q

    # 1. Count the connections the agent opens.
    opened: list[str] = []

    def connect(config, extensions=None):
        opened.append(config.server_id)
        return Client(server)

    monkeypatch.setattr(module, "create_mcp_client", connect)
    config = StdioMCPServerConfig(
        server_id="docs", command="unused",
        tool_approval_modes={"search": ToolApprovalMode.AUTO_APPROVED},
    )
    asks = AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id="s1", tool_name="docs_search", parameters={"q": "mcp"})])
    agent = Agent("test", "test", "test", FakeClient(asks, say("listo"), say("otra")), mcp=[config])

    # 2. Two runs use the MCP tool over the same connection.
    for task in ("busca", "otra vez"):
        async for _ in agent.run_stream_events(task, RunContext(user_id="u")):
            pass
    assert searched == ["mcp"] and opened == ["docs"]

    # 3. close() releases it.
    await agent.close()
    assert agent._mcp.get_tools() == []
