from .capability import CoreAgentCapabilities
from .compaction import CompactionResult, CoreCompaction
from .completion_gate import CompletionCheck, CompletionDecision, CompletionGate
from .component import Component, ComponentBase, CoreLifecycleComponent
from .context import CoreLogBookRegistry
from .embeddings import (
    DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
    get_lightweight_embedding,
    get_lightweight_embeddings,
)
from .knowledge import CoreKnowledgeRegistry
from .layer import CoreLayer
from .memory import CoreMemoryRegistry
from .middleware import CoreMiddleware
from .observation import ObservationRecord
from .routines import CoreRoutineRegistry
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
    "CoreRoutineRegistry",
    "CoreLogBookRegistry",
    "DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL",
    "get_lightweight_embedding",
    "get_lightweight_embeddings",
    "ToolContext",
    "CoreSkillRegistry",
    "ObservationRecord",
    "WorkspaceBase",
    "CompletionCheck",
    "CompletionDecision",
    "CompletionGate",
]


def __getattr__(name: str):
    """Lazily expose heavy exports without creating import cycles."""
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
