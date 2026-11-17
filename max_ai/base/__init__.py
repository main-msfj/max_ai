"""Public contracts and base classes for Max AI components.

This package exposes the interfaces used to compose agents and implement
clients, capabilities, tools, middleware, and runtime services.
"""

from .capability import CoreAgentCapabilities
from .completion_gate import (
    CompletionBase,
    CompletionDecision,
)
from .component import Component, ComponentBase, CoreLifecycleComponent, allow_providers
from .knowledge import CoreKnowledgeRegistry
from .layer import CoreLayer
from .middleware import CoreMiddleware, ModelCall
from .quota_store import CoreQuotaStore
from .session_store import CoreSessionStore
from .tools import CoreTool, ToolContext

__all__ = [
    "CoreLayer",
    "Component",
    "allow_providers",
    "ComponentBase",
    "CoreLifecycleComponent",
    "CoreAgentCapabilities",
    "CoreMiddleware",
    "ModelCall",
    "CoreQuotaStore",
    "CoreSessionStore",
    "CoreTool",
    "CoreKnowledgeRegistry",
    "ToolContext",
    "CompletionBase",
    "CompletionDecision",
]

