"""Create MCP SDK v2 clients from serializable server configurations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from ._model import HTTPServerConfig, MCPServerConfig, StdioMCPServerConfig


def create_mcp_client(config: MCPServerConfig) -> Client:
    """The v2 client probes 2026-07-28 and falls back to legacy MCP."""
    if isinstance(config, StdioMCPServerConfig):
        return Client(StdioServerParameters(
            command=config.command, args=config.args, env=config.env or None,
        ))
    if isinstance(config, HTTPServerConfig):
        if config.transport_protocol == "sse":
            return Client(sse_client(config.url, headers=config.request_headers))

        @asynccontextmanager
        async def transport() -> AsyncIterator[object]:
            async with httpx2.AsyncClient(headers=config.request_headers) as http_client:
                async with streamable_http_client(
                    config.url, http_client=http_client
                ) as streams:
                    yield streams

        return Client(transport())
    raise TypeError(f"Unsupported MCP server config: {type(config).__name__}")
