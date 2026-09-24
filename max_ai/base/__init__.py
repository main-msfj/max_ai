"""Public contracts and base classes for Max AI components.

This package exposes the interfaces used to compose agents and implement
clients, capabilities, tools, middleware, and runtime services.
"""

from .capability import CoreAgentCapabilities
from .compaction import CompactionResult, CoreCompaction
from .completion_gate import (
    CompletionBase,
    CompletionCheck,
    CompletionDecision,
)
from .component import Component, ComponentBase, CoreLifecycleComponent, allow_providers
from .context import CoreLogBookRegistry
from .knowledge import CoreKnowledgeRegistry
from .layer import CoreLayer
from .memory import CoreMemoryRegistry
from .middleware import (
    CoreMiddleware,
    MiddlewareContext,
    ModelRequest,
    StopRun,
    ToolRequest,
)
from .quota_store import CoreQuotaStore
from .session_store import CoreSessionStore
from .skills import CoreSkillRegistry
from .tools import CoreRuntimeTool, CoreTool, ToolContext
from .workspace import WorkspaceBase

__all__ = [
    "CoreLayer",
    "Component",
    "allow_providers",
    "ComponentBase",
    "CoreLifecycleComponent",
    "CompactionResult",
    "CoreCompaction",
    "CoreAgentCapabilities",
    "CoreMiddleware",
    "MiddlewareContext",
    "ModelRequest",
    "StopRun",
    "ToolRequest",
    "CoreQuotaStore",
    "CoreSessionStore",
    "CoreTool",
    "CoreRuntimeTool",
    "CoreMemoryRegistry",
    "CoreKnowledgeRegistry",
    "CoreLogBookRegistry",
    "ToolContext",
    "CoreSkillRegistry",
    "WorkspaceBase",
    "CompletionBase",
    "CompletionCheck",
    "CompletionDecision",
]


def __getattr__(name: str):
    """Lazily expose heavy exports without creating import cycles."""
    if name == "Agent":
        from ..agents.agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
