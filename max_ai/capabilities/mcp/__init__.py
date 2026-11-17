"""MCP integration for MaxAI agents."""

from ._model import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioMCPServerConfig,
    deserialize_mcp_servers,
    ensure_serializable,
    serialize_mcp_servers,
)
from .client_manager import MCPClientManager
from .tool import MCPResourceTool, MCPTool

__all__ = [
    "HTTPServerConfig",
    "MCPClientManager",
    "MCPResourceTool",
    "MCPServerConfig",
    "MCPTool",
    "StdioMCPServerConfig",
    "deserialize_mcp_servers",
    "serialize_mcp_servers",
    "ensure_serializable",
]
