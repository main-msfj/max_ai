"""High-level helpers for creating MCP tools for agents."""

from __future__ import annotations

import typing as t

from ..base.tools import CoreTool
from .client_manager import MCPClientManager
from .config import MCPServerConfig


async def create_mcp_tools(
    server_configs: list[MCPServerConfig],
    auto_connect: bool = True,
) -> tuple[MCPClientManager, list[CoreTool | t.Callable[..., t.Any]]]:
    """Create an MCP manager and discovered CoreTool adapters.

    The returned tools are ready to pass to ``Agent(toolset=...)``. The returned
    manager remains owned by the application so it can close MCP sessions with
    ``disconnect_all()`` when the agent/app is done.
    """
    client_manager = MCPClientManager()
    for config in server_configs:
        client_manager.add_server(config)

    if auto_connect:
        await client_manager.connect_all()

    return client_manager, client_manager.get_tools()
