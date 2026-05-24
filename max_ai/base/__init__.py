from .tools import CoreRuntimeTool, CoreTool, ToolContext
from .layer import CoreLayer
from .memory import CoreMemoryRegistry
from .knowledge import CoreKnowledgeRegistry
from .routines import CoreRoutineRegistry
from .middleware import CoreMiddleware
from .component import Component, ComponentBase, CoreLifecycleComponent
from .compaction import CompactionResult, CoreCompaction
from .capability import CoreAgentCapabilities
from .context import CoreLogBookRegistry
from .embeddings import (
    DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL,
    get_lightweight_embedding,
    get_lightweight_embeddings,
)
from .skills import CoreSkillRegistry

__all__ = [
    "Agent",
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
]


def __getattr__(name: str):
    """Lazily expose heavy exports without creating import cycles."""
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
