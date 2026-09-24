from __future__ import annotations

import pytest
from mcp import Client
from mcp.server import MCPServer

from max_ai.agents import Agent as StackAgent
from max_ai.base.agent import Agent as BaseAgent
from max_ai.capabilities.mcp import MCPClientManager, StdioMCPServerConfig
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


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

    monkeypatch.setattr(module, "create_mcp_client", lambda config: Client(servers[config.server_id]))
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
@pytest.mark.parametrize("agent_class", [BaseAgent, StackAgent])
async def test_agent_accepts_mcp_separately_and_cleans_up(monkeypatch, agent_class):
    from max_ai.capabilities.mcp import client_manager as module

    server = MCPServer("docs")

    @server.tool()
    async def search(q: str) -> str:
        return q

    monkeypatch.setattr(module, "create_mcp_client", lambda config: Client(server))
    config = StdioMCPServerConfig(server_id="docs", command="unused")
    agent = agent_class(
        name="test", description="test", instructions="test", client=None,
        toolset=[], mcp=[config],
    )
    observed = []

    async def fake_drive(*args):
        tool = agent._registry.get("docs_search")
        observed.append(tool is not None and agent._registry.runs_on_host("docs_search"))
        return "finished"

    monkeypatch.setattr(agent, "_drive_connected", fake_drive)
    assert await agent._drive(None, None, None, False, {}) == "finished"
    assert observed == [True]
    if agent_class is BaseAgent:  # the old Agent connects and disconnects per run
        assert agent._registry.get("docs_search") is None
    else:  # one connection shared by every run, released by close()
        assert await agent._drive(None, None, None, False, {}) == "finished"
        assert observed == [True, True]
        assert len(agent._mcp_manager.get_tools()) == 1
        await agent.close()
        assert all(s.worker is None for s in agent._mcp_manager._servers.values())
    assert agent.mcp_servers[0] == StdioMCPServerConfig.model_validate_json(
        agent.mcp_servers[0].model_dump_json()
    )
