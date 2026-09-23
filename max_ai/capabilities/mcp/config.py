"""Compatibility imports for MCP configuration models."""

from ._model import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
    deserialize_mcp_servers,
    ensure_serializable,
    serialize_mcp_servers,
)

__all__ = [
    "HTTPServerConfig", "MCPServerConfig", "StdioMCPServerConfig",
    "serialize_mcp_servers", "ensure_serializable", "deserialize_mcp_servers",
]
