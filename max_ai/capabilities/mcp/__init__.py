"""MCP integration for MaxAI agents."""

from .client_manager import MCPClientManager
from .config import HTTPServerConfig, MCPServerConfig, StdioMCPServerConfig
from .integration import create_mcp_tools
from .tool import MCPResourceTool, MCPTool

__all__ = [
    "HTTPServerConfig",
    "MCPClientManager",
    "MCPResourceTool",
    "MCPServerConfig",
    "MCPTool",
    "StdioMCPServerConfig",
    "create_mcp_tools",
]
