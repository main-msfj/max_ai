"""Lifecycle manager for MCP SDK v2 clients and discovered tools."""

from __future__ import annotations

import asyncio
import typing as t
from dataclasses import dataclass, field

from mcp import Client, MCPError

from ...base.tools import CoreTool
from ...errors.mcp import MCPServerNotFoundError, MCPServerRegistrationError
from ...types.tools import ToolApprovalMode
from ._model import MCPServerConfig
from .tool import MCPResourceTool, MCPTool
from .transport import create_mcp_client


@dataclass
class _ManagedServer:
    config: MCPServerConfig
    worker: asyncio.Task[None] | None = None
    requests: asyncio.Queue[t.Any] = field(default_factory=asyncio.Queue)
    tools: list[CoreTool] = field(default_factory=list)


class MCPClientManager:
    """Own MCP clients in dedicated tasks so SDK calls share their lifecycle."""

    def __init__(self) -> None:
        self._servers: dict[str, _ManagedServer] = {}

    @property
    def server_ids(self) -> list[str]:
        return list(self._servers)

    def add_server(self, config: MCPServerConfig) -> None:
        if not isinstance(config, MCPServerConfig) or type(config) is MCPServerConfig:
            raise TypeError("Use StdioMCPServerConfig or HTTPServerConfig")
        if config.server_id in self._servers:
            raise MCPServerRegistrationError(
                f"MCP server already registered: {config.server_id}"
            )
        self._servers[config.server_id] = _ManagedServer(config=config)

    async def connect(self, server_id: str) -> None:
        server = self._get_server(server_id)
        if server.worker is not None:
            if not server.worker.done():
                return
            server.worker = None  # the connection died: open a new one
        ready = asyncio.get_running_loop().create_future()
        server.worker = asyncio.create_task(self._serve(server, ready))
        try:
            server.tools = await ready
        except BaseException:
            await self.disconnect(server_id)
            raise

    async def _serve(self, server: _ManagedServer, ready: asyncio.Future) -> None:
        try:
            async with create_mcp_client(server.config) as client:
                ready.set_result(await self._discover_server_tools(server.config, client))
                while True:
                    request = await server.requests.get()
                    if request is None:
                        break
                    method, args, future = request
                    if future.done():
                        continue
                    try:
                        future.set_result(await getattr(client, method)(*args))
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(exc)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            while not server.requests.empty():
                request = server.requests.get_nowait()
                if request is not None:
                    future = request[2]
                    if not future.done():
                        future.set_exception(RuntimeError("MCP connection closed"))

    async def connect_all(self) -> None:
        try:
            for server_id in self.server_ids:
                await self.connect(server_id)
        except BaseException:
            await self.disconnect_all()
            raise

    def get_tools(self) -> list[CoreTool]:
        return [tool for server in self._servers.values() for tool in server.tools]

    async def disconnect(self, server_id: str) -> None:
        server = self._get_server(server_id)
        worker = server.worker
        server.worker = None
        server.tools = []
        if worker is not None:
            await server.requests.put(None)
            await worker

    async def disconnect_all(self) -> None:
        for server_id in reversed(self.server_ids):
            await self.disconnect(server_id)

    async def call_tool(
        self, server_id: str, tool_name: str,
        arguments: dict[str, t.Any], timeout_seconds: float,
    ) -> t.Any:
        return await self._request(server_id, "call_tool", tool_name, arguments, timeout_seconds)

    async def read_resource(self, server_id: str, uri: str) -> t.Any:
        return await self._request(server_id, "read_resource", uri)

    async def _request(self, server_id: str, method: str, *args: t.Any) -> t.Any:
        server = self._get_server(server_id)
        if server.worker is None:
            await self.connect(server_id)
        elif server.worker.done():
            await server.worker
            raise RuntimeError(f"MCP connection closed: {server_id}")
        future = asyncio.get_running_loop().create_future()
        await server.requests.put((method, args, future))
        return await future

    def _get_server(self, server_id: str) -> _ManagedServer:
        try:
            return self._servers[server_id]
        except KeyError as exc:
            raise MCPServerNotFoundError(f"Unknown MCP server: {server_id}") from exc

    async def _discover_server_tools(
        self, config: MCPServerConfig, client: Client,
    ) -> list[CoreTool]:
        discovered: list[CoreTool] = []
        for tool_def in await self._list_all_tools(client):
            discovered.append(MCPTool(
                mcp_tool_name=tool_def.name,
                mcp_tool_description=tool_def.description or tool_def.name,
                mcp_tool_schema=tool_def.input_schema,
                client_manager=self,
                server_id=config.server_id,
                approval_mode=self._approval_mode_for_tool(config, tool_def),
                timeout_seconds=config.timeout_seconds,
            ))
        resources, templates = await self._list_resources(client)
        if resources or templates:
            discovered.append(MCPResourceTool(
                client_manager=self,
                server_id=config.server_id,
                available_resources=resources,
                resource_templates=templates,
                approval_mode=config.effective_resource_approval_mode,
                timeout_seconds=config.timeout_seconds,
            ))
        return discovered

    @staticmethod
    def _approval_mode_for_tool(config: MCPServerConfig, tool_def: t.Any) -> ToolApprovalMode:
        explicit = config.tool_approval_modes.get(tool_def.name)
        if explicit is not None:
            return explicit
        annotations = getattr(tool_def, "annotations", None)
        if annotations is None:
            return config.approval_mode
        if (annotations.read_only_hint is True
                and annotations.destructive_hint is not True
                and annotations.open_world_hint is not True):
            return ToolApprovalMode.AUTO_APPROVED
        if (annotations.destructive_hint is True
                or annotations.open_world_hint is True
                or annotations.read_only_hint is False):
            return ToolApprovalMode.ASK_APPROVED
        return config.approval_mode

    @staticmethod
    async def _list_all_tools(client: Client) -> list[t.Any]:
        tools: list[t.Any] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools

    @staticmethod
    async def _list_resources(client: Client) -> tuple[list[t.Any], list[t.Any]]:
        resources: list[t.Any] = []
        templates: list[t.Any] = []
        try:
            cursor: str | None = None
            while True:
                page = await client.list_resources(cursor=cursor)
                resources.extend(page.resources)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except MCPError:
            pass
        try:
            cursor = None
            while True:
                page = await client.list_resource_templates(cursor=cursor)
                templates.extend(page.resource_templates)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except MCPError:
            pass
        return resources, templates
