"""Compatibility imports for MCP configuration models."""

from ._model import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
    deserialize_mcp_servers,
    serialize_mcp_servers,
)

__all__ = [
    "HTTPServerConfig", "MCPServerConfig", "StdioMCPServerConfig",
    "serialize_mcp_servers", "deserialize_mcp_servers",
]
