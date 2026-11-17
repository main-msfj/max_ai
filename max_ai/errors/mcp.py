"""Errors for MCP integration."""

import time


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


class MCPServerUnavailableError(MCPToolError):
    """An MCP server could not be reached; its tools fail until it is back."""


class MCPConnectionLost(MCPError, ConnectionError):
    """The connection to the MCP server closed while a request waited.

    The request may or may not have run on the server: only read-only calls
    are sent again (after reconnecting); others go back to the model.
    """


class MCPTaskRunning(MCPToolError):
    """The call became a task that did not end within the wait."""

    def __init__(self, task_id: str, ttl_ms: int | None) -> None:
        super().__init__(f"Task {task_id} is still running.")
        self.task_id = task_id
        self.expires_at = None if ttl_ms is None else time.time() + ttl_ms / 1000
