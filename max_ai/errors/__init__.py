"""Every exception of max_ai, in one place."""

from .client import ClientError
from .context import ChatHistoryError
from .embeddings import EmbeddingError
from .harness import HarnessError
from .mcp import (
    MCPConnectionLost,
    MCPError,
    MCPServerConfigError,
    MCPServerNotFoundError,
    MCPServerRegistrationError,
    MCPServerUnavailableError,
    MCPTaskRunning,
    MCPToolContentError,
    MCPToolError,
    MCPToolExecutionError,
)
from .middleware import StopRun
from .sandbox import SandboxError
from .skills import SkillError
from .stacks import StackError
from .tools import ToolDefinitionError, ToolRegistryError, ToolRetry

__all__ = [
    "ChatHistoryError",
    "ClientError",
    "EmbeddingError",
    "HarnessError",
    "MCPConnectionLost",
    "MCPError",
    "MCPServerConfigError",
    "MCPServerNotFoundError",
    "MCPServerRegistrationError",
    "MCPServerUnavailableError",
    "MCPTaskRunning",
    "MCPToolContentError",
    "MCPToolError",
    "MCPToolExecutionError",
    "SandboxError",
    "SkillError",
    "StackError",
    "StopRun",
    "ToolDefinitionError",
    "ToolRegistryError",
    "ToolRetry",
]
