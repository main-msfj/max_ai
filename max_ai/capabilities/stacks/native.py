"""NativePrompt: the system prompt of an agent, from the pieces it has."""

from __future__ import annotations

import typing as t

from ...base.layer import CoreLayer
from ...core.tool.registry import ToolRegistry
from .agent_policy_layer import AgentPolicyLayer
from .builder import PromptBuilder
from .custom_layer import CustomLayer
from .knowledge_layer import KnowledgeLayer
from .memory_files_layer import MemoryFilesLayer
from .rendering_layer import RenderingLayer
from .session_layer import SessionLayer
from .skills_layer import SkillsLayer
from .task_analysis_layer import TaskAnalysisLayer
from .tools_layer import ToolsLayer
from .workspace_layer import WorkspaceLayer

if t.TYPE_CHECKING:
    from ...base.sandbox import SandboxCore
    from ...base.skills import SkillsCore
    from ..memory import Memory


class NativePrompt:
    """Knows which layer goes with each piece of an agent. The agent says
    what it has; the builder, which knows no layer, puts them in order."""

    # From what never changes to what changes during a conversation, so the
    # provider's cache keeps most of the prompt.
    ORDER: t.ClassVar[list[type[CoreLayer]]] = [
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

    @classmethod
    async def build(
        cls,
        registry: ToolRegistry,
        variables: dict[str, t.Any],  # name, description, instructions, current_date
        *,
        session: str,
        sandbox: SandboxCore | None = None,  # already bound to the user
        skills: SkillsCore | None = None,
        memory: Memory | None = None,
        memory_index: str = "",  # as the model saw it when the session opened
        knowledge: bool = False,
        extra: t.Sequence[CoreLayer] = (),  # the developer's own layers
    ) -> str:
        """The native layers always, each optional one only when the agent
        has its piece, then the developer's, all in the standard order."""
        layers: list[CoreLayer] = [
            AgentPolicyLayer(), TaskAnalysisLayer(), RenderingLayer(), ToolsLayer(), SessionLayer(),
        ]
        variables = dict(variables)

        # 1. With a sandbox: where the session works.
        if sandbox is not None:
            layers.append(WorkspaceLayer())
            variables.update(sandbox.session_paths(session))

        # 2. The skills the agent was given that are installed there.
        if sandbox is not None and skills is not None:
            layers.append(SkillsLayer())
            installed = await sandbox.installed_skills()
            variables["skills"] = [
                {"name": name, "description": installed[name]["description"]}
                for name in skills.skills
                if name in installed
            ]

        # 3. The memory, with its index.
        if sandbox is not None and memory is not None:
            layers.append(MemoryFilesLayer())
            variables["memory"] = memory.block(sandbox, memory_index)

        # 4. Knowledge to search.
        if knowledge:
            layers.append(KnowledgeLayer())

        return PromptBuilder([*layers, *extra], registry, order=cls.ORDER).build(variables)


__all__ = ["NativePrompt"]
