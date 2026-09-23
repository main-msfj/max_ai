from .capability import CoreAgentCapabilities
from .compaction import CompactionResult, CoreCompaction
from .completion_gate import (
    CompletionBase,
    CompletionCheck,
    CompletionDecision,
)
from .component import Component, ComponentBase, CoreLifecycleComponent
from .context import CoreLogBookRegistry
from .knowledge import CoreKnowledgeRegistry
from .layer import CoreLayer
from .memory import CoreMemoryRegistry
from .middleware import CoreMiddleware
from .skills import CoreSkillRegistry
from .tools import CoreRuntimeTool, CoreTool, ToolContext
from .workspace import WorkspaceBase

__all__ = [
    "CoreLayer",
    "Component",
    "ComponentBase",
    "CoreLifecycleComponent",
    "CompactionResult",
    "CoreCompaction",
    "CoreAgentCapabilities",
    "CoreMiddleware",
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
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
