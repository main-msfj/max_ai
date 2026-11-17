"""Module providing capability implementations and supporting utilities."""
from ...base.layer import CoreLayer
from .agent_policy_layer import AgentPolicyLayer
from .builder import PromptBuilder
from .custom_layer import CustomLayer
from .knowledge_layer import KnowledgeLayer
from .memory_files_layer import MemoryFilesLayer
from .memory_layer import MemoryLayer
from .rendering_layer import RenderingLayer
from .session_layer import SessionLayer
from .session_state_layer import SessionStateLayer
from .skills_layer import SkillsLayer
from .task_analysis_layer import TaskAnalysisLayer
from .tools_layer import ToolsLayer
from .workspace_layer import WorkspaceLayer

# From what never changes to what changes during a conversation, so the
# provider's cache keeps most of the prompt. The harness will own this.
STANDARD_ORDER: list[type[CoreLayer]] = [
    TaskAnalysisLayer,
    RenderingLayer,
    AgentPolicyLayer,
    CustomLayer,
    ToolsLayer,
    SkillsLayer,
    KnowledgeLayer,
    WorkspaceLayer,
    MemoryFilesLayer,
    SessionLayer,
]

__all__ = [
    "STANDARD_ORDER",
    "CoreLayer",
    "AgentPolicyLayer",
    "CustomLayer",
    "MemoryLayer",
    "MemoryFilesLayer",
    "KnowledgeLayer",
    "PromptBuilder",
    "RenderingLayer",
    "SessionLayer",
    "SessionStateLayer",
    "TaskAnalysisLayer",
    "SkillsLayer",
    "ToolsLayer",
    "WorkspaceLayer",
]
