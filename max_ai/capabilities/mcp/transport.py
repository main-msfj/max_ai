"""Transport helpers for MCP server connections."""

from __future__ import annotations

import logging
import typing as t
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import httpx
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from ...loggers import ScopedLogger
from .config import HTTPServerConfig, MCPServerConfig, StdioMCPServerConfig

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="mcp.transport")


@dataclass
class MCPTransportConnection:
    """Open MCP transport streams plus closeable context state."""

    read: t.Any
    write: t.Any
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def close(self) -> None:
        # AsyncExitStack unwinds every entered context in LIFO order, in the
        # same task that entered them. This is required for the anyio
        # task-group-backed transport contexts; unwinding out of order or in a
        # different task triggers "Cancelled via cancel scope" errors.
        await self.exit_stack.aclose()


async def _connect_stdio_server(config: StdioMCPServerConfig) -> MCPTransportConnection:
    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env or None,
    )
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(stdio_client(params))
        return MCPTransportConnection(read=read, write=write, exit_stack=stack)
    except BaseException:
        await stack.aclose()
        raise


async def _connect_streamable_http_server(
    config: HTTPServerConfig,
) -> MCPTransportConnection:
    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(headers=config.headers)
        )
        read, write, _get_session_id = await stack.enter_async_context(
            streamable_http_client(url=config.url, http_client=http_client)
        )
        return MCPTransportConnection(read=read, write=write, exit_stack=stack)
    except BaseException:
        await stack.aclose()
        raise


async def _connect_sse_server(config: HTTPServerConfig) -> MCPTransportConnection:
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(
            sse_client(url=config.url, headers=config.headers)
        )
        return MCPTransportConnection(read=read, write=write, exit_stack=stack)
    except BaseException:
        await stack.aclose()
        raise


async def connect_to_mcp_server(config: MCPServerConfig) -> MCPTransportConnection:
    """Open transport streams for a configured MCP server."""
    try:
        if isinstance(config, StdioMCPServerConfig):
            return await _connect_stdio_server(config)
        if isinstance(config, HTTPServerConfig):
            if config.transport_protocol == "sse":
                return await _connect_sse_server(config)
            return await _connect_streamable_http_server(config)
    except Exception as exc:
        log.error("Error connecting to MCP server", server_id=config.server_id, exc=exc)
        raise

    msg = f"Unsupported MCP server config: {type(config).__name__}"
    log.error(msg, server_id=config.server_id)
    raise ValueError(msg)
