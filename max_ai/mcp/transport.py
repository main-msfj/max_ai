"""Transport helpers for MCP server connections."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import httpx
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from ..loggers import ScopedLogger
from .config import HTTPServerConfig, MCPServerConfig, StdioMCPServerConfig

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="mcp.transport")


@dataclass
class MCPTransportConnection:
    """Open MCP transport streams plus closeable context state."""

    read: t.Any
    write: t.Any
    client_ctx: t.Any
    http_client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        try:
            await self.client_ctx.__aexit__(None, None, None)
        finally:
            if self.http_client is not None:
                await self.http_client.aclose()


async def _connect_stdio_server(config: StdioMCPServerConfig) -> MCPTransportConnection:
    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env or None,
    )
    client_ctx = stdio_client(params)
    read, write = await client_ctx.__aenter__()
    return MCPTransportConnection(read=read, write=write, client_ctx=client_ctx)


async def _connect_streamable_http_server(
    config: HTTPServerConfig,
) -> MCPTransportConnection:
    http_client = httpx.AsyncClient(headers=config.headers)
    client_ctx = streamable_http_client(url=config.url, http_client=http_client)
    try:
        read, write, _get_session_id = await client_ctx.__aenter__()
        return MCPTransportConnection(
            read=read,
            write=write,
            client_ctx=client_ctx,
            http_client=http_client,
        )
    except Exception:
        await http_client.aclose()
        raise


async def _connect_sse_server(config: HTTPServerConfig) -> MCPTransportConnection:
    client_ctx = sse_client(url=config.url, headers=config.headers)
    read, write = await client_ctx.__aenter__()
    return MCPTransportConnection(read=read, write=write, client_ctx=client_ctx)


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
