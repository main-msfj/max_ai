"""max_ai: framework for production LLM agents.

The common entry points are re-exported here::

    from max_ai import Agent, OpenAIChatCompletionClient, tool

All re-exports are lazy (resolved on first attribute access), so
``import max_ai`` stays cheap and subpackages keep their import order.
"""

import importlib
import typing as t
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maxai")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Agent",
    "AgentResponse",
    "FunctionAsTool",
    "OllamaChatCompletionClient",
    "OpenAIChatCompletionClient",
    "OpenRouterChatCompletionClient",
    "RunContext",
    "ToolApprovalMode",
    "tool",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Agent": ("max_ai.agents", "Agent"),
    "AgentResponse": ("max_ai.types.agent_response", "AgentResponse"),
    "FunctionAsTool": ("max_ai.capabilities.tools.function_as_tool", "FunctionAsTool"),
    "OllamaChatCompletionClient": (
        "max_ai.capabilities.clients.ollama", "OllamaChatCompletionClient",
    ),
    "OpenAIChatCompletionClient": (
        "max_ai.capabilities.clients.openai", "OpenAIChatCompletionClient",
    ),
    "OpenRouterChatCompletionClient": (
        "max_ai.capabilities.clients.openrouter", "OpenRouterChatCompletionClient",
    ),
    "RunContext": ("max_ai.types.run_context", "RunContext"),
    "ToolApprovalMode": ("max_ai.types.tools", "ToolApprovalMode"),
    "tool": ("max_ai.capabilities.tools.decorator", "tool"),
}

if t.TYPE_CHECKING:  # static analyzers see the real symbols
    from .agents import Agent
    from .capabilities.clients.ollama import OllamaChatCompletionClient
    from .capabilities.clients.openai import OpenAIChatCompletionClient
    from .capabilities.clients.openrouter import OpenRouterChatCompletionClient
    from .capabilities.tools.decorator import tool
    from .capabilities.tools.function_as_tool import FunctionAsTool
    from .types.agent_response import AgentResponse
    from .types.run_context import RunContext
    from .types.tools import ToolApprovalMode


def __getattr__(name: str) -> t.Any:
    """Resolve a lazily exported attribute."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    return getattr(importlib.import_module(module_name), attr)
