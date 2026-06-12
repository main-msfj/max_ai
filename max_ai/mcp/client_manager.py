"""Lifecycle manager for MCP client sessions and discovered tools."""

from __future__ import annotations

import logging
import typing as t
from contextlib import suppress
from dataclasses import dataclass, field

from mcp import ClientSession, McpError

from ..base.tools import CoreTool
from ..errors.mcp import (
    MCPServerNotFoundError,
    MCPServerRegistrationError,
)
from ..loggers import ScopedLogger
from ..types.tools import ToolApprovalMode
from .config import MCPServerConfig
from .tool import MCPResourceTool, MCPTool
from .transport import MCPTransportConnection, connect_to_mcp_server

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="mcp.manager")


@dataclass
class _ManagedServer:
    config: MCPServerConfig
    transport: MCPTransportConnection | None = None
    session: ClientSession | None = None
    tools: list[CoreTool] = field(default_factory=list)
    connected: bool = False


class MCPClientManager:
    """Own MCP configs, sessions, discovery, and shutdown."""

    def __init__(self) -> None:
        self._servers: dict[str, _ManagedServer] = {}

    @property
    def server_ids(self) -> list[str]:
        return list(self._servers)

    def add_server(self, config: MCPServerConfig) -> None:
        if config.server_id in self._servers:
            raise MCPServerRegistrationError(
                f"MCP server already registered: {config.server_id}"
            )
        self._servers[config.server_id] = _ManagedServer(config=config)

    async def connect(self, server_id: str) -> ClientSession:
        server = self._get_server(server_id)
        if server.connected and server.session is not None:
            return server.session

        logger_for_server = log.child(server_id=server_id)
        logger_for_server.info("Connecting MCP server")
        transport = await connect_to_mcp_server(server.config)
        session = ClientSession(transport.read, transport.write)

        try:
            await session.__aenter__()
            await session.initialize()
            server.transport = transport
            server.session = session
            server.connected = True
            server.tools = await self._discover_server_tools(server)
            logger_for_server.info(
                "MCP server connected",
                tool_count=len(server.tools),
            )
            return session
        except Exception:
            with suppress(Exception):
                await session.__aexit__(None, None, None)
            with suppress(Exception):
                await transport.close()
            server.transport = None
            server.session = None
            server.connected = False
            server.tools = []
            raise

    async def connect_all(self) -> None:
        for server_id in self.server_ids:
            await self.connect(server_id)

    async def get_session(self, server_id: str) -> ClientSession:
        server = self._get_server(server_id)
        if server.session is None or not server.connected:
            return await self.connect(server_id)
        return server.session

    def get_tools(self) -> list[CoreTool]:
        tools: list[CoreTool] = []
        for server in self._servers.values():
            tools.extend(server.tools)
        return tools

    async def disconnect(self, server_id: str) -> None:
        server = self._get_server(server_id)
        session, transport = server.session, server.transport
        server.session = None
        server.transport = None
        server.connected = False
        server.tools = []

        if session is not None:
            with suppress(Exception):
                await session.__aexit__(None, None, None)
        if transport is not None:
            with suppress(Exception):
                await transport.close()

    async def disconnect_all(self) -> None:
        for server_id in list(self._servers):
            await self.disconnect(server_id)

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, t.Any],
        timeout_seconds: float,
    ) -> t.Any:
        session = await self.get_session(server_id)
        return await session.call_tool(
            tool_name,
            arguments,
            read_timeout_seconds=self._timeout_delta(timeout_seconds),
        )

    async def read_resource(self, server_id: str, uri: t.Any) -> t.Any:
        session = await self.get_session(server_id)
        return await session.read_resource(uri)

    def _get_server(self, server_id: str) -> _ManagedServer:
        try:
            return self._servers[server_id]
        except KeyError as exc:
            raise MCPServerNotFoundError(f"Unknown MCP server: {server_id}") from exc

    async def _discover_server_tools(self, server: _ManagedServer) -> list[CoreTool]:
        assert server.session is not None
        config = server.config
        discovered: list[CoreTool] = []

        tool_defs = await self._list_all_tools(server.session)
        for tool_def in tool_defs:
            discovered.append(
                MCPTool(
                    mcp_tool_name=tool_def.name,
                    mcp_tool_description=tool_def.description or tool_def.name,
                    mcp_tool_schema=tool_def.inputSchema,
                    client_manager=self,
                    server_id=config.server_id,
                    approval_mode=self._approval_mode_for_tool(config, tool_def),
                    timeout_seconds=config.timeout_seconds,
                )
            )

        resources, templates = await self._list_resources(server.session)
        if resources or templates:
            discovered.append(
                MCPResourceTool(
                    client_manager=self,
                    server_id=config.server_id,
                    available_resources=resources,
                    resource_templates=templates,
                    approval_mode=config.resource_approval_mode,
                    timeout_seconds=config.timeout_seconds,
                )
            )

        return discovered

    @staticmethod
    def _approval_mode_for_tool(
        config: MCPServerConfig,
        tool_def: t.Any,
    ) -> ToolApprovalMode:
        explicit = config.tool_approval_modes.get(tool_def.name)
        if explicit is not None:
            return explicit

        annotations = getattr(tool_def, "annotations", None)
        if annotations is None:
            return config.approval_mode

        is_read_only = annotations.readOnlyHint is True
        is_destructive = annotations.destructiveHint is True
        is_open_world = annotations.openWorldHint is True
        if is_read_only and not is_destructive and not is_open_world:
            return ToolApprovalMode.AUTO_APPROVED
        if is_destructive or is_open_world or annotations.readOnlyHint is False:
            return ToolApprovalMode.ASK_APPROVED
        return config.approval_mode

    @staticmethod
    async def _list_all_tools(session: ClientSession) -> list[t.Any]:
        tools: list[t.Any] = []
        cursor: str | None = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                return tools

    @staticmethod
    async def _list_resources(session: ClientSession) -> tuple[list[t.Any], list[t.Any]]:
        resources: list[t.Any] = []
        templates: list[t.Any] = []

        with suppress(McpError):
            cursor: str | None = None
            while True:
                result = await session.list_resources(cursor=cursor)
                resources.extend(result.resources)
                cursor = result.nextCursor
                if cursor is None:
                    break

        with suppress(McpError):
            cursor = None
            while True:
                result = await session.list_resource_templates(cursor=cursor)
                templates.extend(result.resourceTemplates)
                cursor = result.nextCursor
                if cursor is None:
                    break

        return resources, templates

    @staticmethod
    def _timeout_delta(timeout_seconds: float) -> t.Any:
        from datetime import timedelta

        return timedelta(seconds=timeout_seconds)
