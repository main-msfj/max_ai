"""An unreachable MCP server is bounded in time and never takes the others down."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from mcp import Client
from mcp.server import MCPServer

from max_ai.capabilities.mcp import MCPClientManager, StdioMCPServerConfig
from max_ai.errors.mcp import MCPServerUnavailableError
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


def echo_server(name: str) -> MCPServer:
    server = MCPServer(name)

    @server.tool(name="echo")
    async def echo(value: str) -> str:
        return value

    return server


@contextlib.asynccontextmanager
async def hangs():
    await asyncio.Event().wait()
    yield


@contextlib.asynccontextmanager
async def refuses():
    raise ConnectionError("connection refused")
    yield


def manager_with(monkeypatch, clients, **config):
    from max_ai.capabilities.mcp import client_manager as module

    attempts = {server_id: 0 for server_id in clients}

    def create(server_config):
        attempts[server_config.server_id] += 1
        return clients[server_config.server_id]()

    monkeypatch.setattr(module, "create_mcp_client", create)
    manager = MCPClientManager()
    for server_id in clients:
        manager.add_server(StdioMCPServerConfig(
            server_id=server_id, command="unused",
            tool_approval_modes={"echo": ToolApprovalMode.AUTO_APPROVED},
            **config.get(server_id, {}),
        ))
    return manager, attempts


async def test_a_hanging_server_times_out_and_the_others_still_connect(monkeypatch):
    good = echo_server("good")
    manager, _ = manager_with(
        monkeypatch, {"good": lambda: Client(good), "slow": hangs},
        slow={"connect_timeout_seconds": 0.2},
    )
    try:
        unavailable = await asyncio.wait_for(manager.connect_all(), 5)
        assert list(unavailable) == ["slow"]
        assert "no answer within 0.2s" in str(unavailable["slow"])
        assert [tool.name for tool in manager.get_tools()] == ["good_echo"]
    finally:
        await manager.disconnect_all()


async def test_a_failing_server_keeps_healthy_connections_open(monkeypatch):
    good = echo_server("good")
    clients = {"good": lambda: Client(good), "flaky": lambda: Client(echo_server("flaky"))}
    manager, _ = manager_with(monkeypatch, clients, flaky={"retry_after_seconds": 0})
    try:
        assert await manager.connect_all() == {}
        worker = manager._servers["good"].worker
        flaky_echo = next(t for t in manager.get_tools() if t.server_id == "flaky")

        # flaky drops and cannot come back; good is untouched.
        await manager.disconnect("flaky")
        manager._servers["flaky"].tools = [flaky_echo]
        clients["flaky"] = refuses
        unavailable = await manager.connect_all()
        assert list(unavailable) == ["flaky"]
        assert manager._servers["good"].worker is worker and not worker.done()

        # Its known tools stay registered and answer that it is unavailable.
        result = await flaky_echo.execute(ToolCallRecord(
            tool_name=flaky_echo.name, parameters={"value": "x"},
        ))
        assert result.success is False and "unavailable" in result.error
    finally:
        await manager.disconnect_all()


async def test_a_failed_server_is_not_retried_until_retry_after(monkeypatch):
    manager, attempts = manager_with(
        monkeypatch, {"down": refuses}, down={"retry_after_seconds": 60},
    )
    for _ in range(3):
        assert list(await manager.connect_all()) == ["down"]
    assert attempts["down"] == 1
    manager._servers["down"].failed_at -= 61
    await manager.connect_all()
    assert attempts["down"] == 2


async def test_a_required_server_fails_the_run(monkeypatch):
    manager, _ = manager_with(monkeypatch, {"must": refuses}, must={"required": True})
    with pytest.raises(MCPServerUnavailableError, match="connection refused"):
        await manager.connect_all()


async def test_concurrent_callers_share_one_attempt(monkeypatch):
    manager, attempts = manager_with(
        monkeypatch, {"slow": hangs}, slow={"connect_timeout_seconds": 0.2},
    )
    results = await asyncio.wait_for(
        asyncio.gather(*(manager.connect("slow") for _ in range(5)), return_exceptions=True), 5,
    )
    assert all(isinstance(r, MCPServerUnavailableError) for r in results)
    assert attempts["slow"] == 1
