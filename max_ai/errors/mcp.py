"""Errors for MCP integration."""


class MCPError(Exception):
    """Base exception for MCP integration errors."""


class MCPServerConfigError(MCPError, ValueError):
    """Raised when an MCP server configuration is invalid."""


class MCPServerRegistrationError(MCPError, ValueError):
    """Raised when MCP server registration fails."""


class MCPServerNotFoundError(MCPError, KeyError):
    """Raised when an MCP server id is not registered."""


class MCPToolError(MCPError):
    """Base exception for MCP tool adapter errors."""


class MCPToolExecutionError(MCPToolError):
    """The remote MCP tool returned an error result."""


class MCPToolContentError(MCPToolError):
    """The remote MCP tool returned content this adapter cannot expose."""
