"""
Builds the variables dict for each prompt layer at agent prepare-time.

For every layer the framework knows about, this class has a method that
pulls the right pieces from the agent's capabilities and returns a dict
matching that layer's declared contract (``_required_variables`` /
``_optional_variables``).

Unknown layer types — e.g. user-provided custom layers — get an empty
dict back. Those layers render with their ``extra_variables`` only.

The whole class is async because some collectors await on the registry
(``memory.get_context()``, ``context.get_current_session_summary()``).
"""

from __future__ import annotations

import typing as t
from collections.abc import Callable, Awaitable

from ..base.layer import CoreLayer

from ..stacks.memory_layer import MemoryLayer
from ..stacks.skills_layer import SkillsLayer
from ..stacks.routine_layer import RoutineLayer
from ..stacks.context_layer import ContextLayer
from ..stacks.knowledge_layer import KnowledgeLayer
from ..stacks.agent_policy_layer import AgentPolicyLayer
from ..stacks.priority_tools_layer import PriorityToolsLayer
from ..base.agent import Agent


CollectorFn = Callable[["PromptVariablesBuilder"], Awaitable[dict[str, t.Any]]]


class PromptVariablesBuilder:
    """Resolves layer-specific variables from the agent at prepare-time.

    One collector method per known framework layer. Layers whose type is
    not in ``_dispatch`` get ``{}`` from ``collect()`` — they render
    with whatever ``extra_variables`` were passed at construction.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def collect(self, layer_type: type[CoreLayer]) -> dict[str, t.Any]:
        """Return the variables dict for the given layer type.

        Returns ``{}`` for unknown types so user-defined layers still
        render via their ``extra_variables`` contract.
        """
        method = self._dispatch.get(layer_type)
        if method is None:
            return {}
        return await method(self)

    # -------- PER-LAYER COLLECTORS -----------------------------------------------------------
    async def _for_agent_policy(self) -> dict[str, t.Any]:
        return {
            "name": self.agent.name,
            "description": self.agent.description,
            "instructions": self.agent.instructions,
        }

    async def _for_memory(self) -> dict[str, t.Any]:
        memory = self.agent.capabilities.memory
        if memory is None:
            return {"persistent_memories": []}

        memories = await memory.get_context()
        result: dict[str, t.Any] = {"persistent_memories": memories}

        tool_names = [t.name for t in self.agent.capabilities.memory_tools]
        if tool_names:
            result["memory_tools"] = tool_names
        return result

    async def _for_knowledge(self) -> dict[str, t.Any]:
        tool_names = [t.name for t in self.agent.capabilities.knowledge_tools]
        if not tool_names:
            return {}
        return {"retrieval_tools": tool_names}

    async def _for_routines(self) -> dict[str, t.Any]:
        tool_names = {t.name for t in self.agent.capabilities.routine_tools}
        result: dict[str, t.Any] = {}
        if "search_routines" in tool_names:
            result["routine_search_tool"] = "search_routines"
        if "get_routine" in tool_names:
            result["routine_fetch_tool"] = "get_routine"
        return result

    async def _for_skills(self) -> dict[str, t.Any]:
        return {"loaded_skills": self.agent.capabilities.loaded_skills}

    async def _for_priority_tools(self) -> dict[str, t.Any]:
        return {"priority_tools": list(self.agent.capabilities.priority_tools)}

    async def _for_context(self) -> dict[str, t.Any]:
        context = self.agent.capabilities.context
        if context is None:
            return {}
        summary = await context.get_current_session_summary()
        return {"current_session_summary": summary}

    # -------- DISPATCH TABLE -----------------------------------------------------------
    # Layers without entries here (RenderingLayer, TaskAnalysisLayer)
    # render with no variables — their templates are static.
    _dispatch: t.ClassVar[dict[type, CollectorFn]] = {
        AgentPolicyLayer: _for_agent_policy,
        MemoryLayer: _for_memory,
        KnowledgeLayer: _for_knowledge,
        RoutineLayer: _for_routines,
        SkillsLayer: _for_skills,
        PriorityToolsLayer: _for_priority_tools,
        ContextLayer: _for_context,
    }
