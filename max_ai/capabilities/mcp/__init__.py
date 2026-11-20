"""MCP integration for MaxAI agents."""

from ._model import (
    HTTPSecret,
    HTTPServerConfig,
    MCPSecretBase,
    MCPServerConfig,
    StdioMCPServerConfig,
    StdioSecret,
    deserialize_mcp_servers,
    serialize_mcp_servers,
)
from .client_manager import MCPClientManager
from .tool import MCPResourceTool, MCPTool

__all__ = [
    "HTTPSecret",
    "HTTPServerConfig",
    "MCPClientManager",
    "MCPResourceTool",
    "MCPSecretBase",
    "MCPServerConfig",
    "MCPTool",
    "StdioMCPServerConfig",
    "StdioSecret",
    "deserialize_mcp_servers",
    "serialize_mcp_servers",
]
