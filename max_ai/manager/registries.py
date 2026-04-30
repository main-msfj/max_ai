"""
Capability Registry — normalizes and validates the agent's runtime
components (tools, memory, knowledge, routines, skills, context) before
the agent is used.

Responsibilities:
  - Coerce heterogeneous inputs to their canonical types.
  - Validate internal coherence (e.g. priority_tools reference real tools).
  - Resolve async capabilities (skills) via ``prepare()``.
  - Provide type-safe lookup APIs for the agent at runtime.

Out of scope:
  - Provider-specific formatting (that lives in each client's
    build_tool_schema / adapt_schema_for_provider).
  - Prompt variable assembly (that lives in PromptVariablesBuilder).
"""

from __future__ import annotations

import logging
import typing as t

from ..errors.capabilities import CapabilityError
from ..loggers import ScopedLogger

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, prefix="[CapabilityRegistry]")

if t.TYPE_CHECKING:
    from ..types.skills import Skill
    from ..base import (
        CoreTool,
        CoreSkillRegistry,
        CoreMemoryRegistry,
        CoreLogBookRegistry,
        CoreRoutineRegistry,
        CoreKnowledgeRegistry,
    )

class CapabilityRegistry:
    """
    Central registry of the agent's capabilities.

    Built in two phases:
      1. ``__init__`` — sync. Normalizes inputs, validates what it can
         see right away (explicit tools, priority_tools, name uniqueness
         across tools/memory/knowledge/routines/context).
      2. ``prepare()`` — async. Loads skills via their registry and
         revalidates only the new tools introduced by skills against
         everything already registered.

    The agent must call ``await registry.prepare()`` before running.

    Cardinality of capabilities:
      - memory, context, skills, routines: ONE registry each (single
        connection per concern).
      - knowledge: a list (multiple sources are common — docs, tickets,
        code, etc. — and their tools have unique names per source).
      - tools: a list of standalone tools.
    """

    def __init__(
        self,
        memory: CoreMemoryRegistry | None = None,
        routines: CoreRoutineRegistry | None = None,
        skills: CoreSkillRegistry | None = None,
        logbook: CoreLogBookRegistry | None = None,
        priority_tools: t.Sequence[str] | None = None,
        toolset: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
        knowledge: t.Sequence[CoreKnowledgeRegistry] | None = None,
    ) -> None:

        self.memory = memory
        self.logbook = logbook
        self.routines = routines
        self.skills_registry = skills
        self.priority_tools: list[str] = list(priority_tools or [])
        self.toolset: list[CoreTool] = self._normalize_tools(toolset or [])
        self.knowledge: list[CoreKnowledgeRegistry] = list(knowledge or [])

        # Populated by prepare(). Empty until then.
        self._loaded_skills: list[Skill] = []
        self._read_resource_tool: CoreTool | None = None
        self._prepared: bool = False

        # Early validation - only covers what's available sync.
        self._validate_priority_tools_exist()
        self._validate_unique_tool_names(self._all_tools_sync())

    # -------- NORMALIZATION ------------------------------------------
    @staticmethod
    def _normalize_tools(
        tools: t.Sequence[CoreTool | t.Callable[..., t.Any]],
    ) -> list[CoreTool]:
        """Coerce a mixed sequence of CoreTool | Callable into CoreTool list."""
        from ..base.tools import CoreTool

        normalized: list[CoreTool] = []
        for tool in tools:
            if isinstance(tool, CoreTool):
                normalized.append(tool)
            elif callable(tool):
                from ..tools.function_as_tool import FunctionAsTool

                normalized.append(FunctionAsTool(tool))
            else:
                raise CapabilityError.invalid_tool_entry(tool)
        return normalized

    # -------- PREPARE ------------------------------------------------
    async def prepare(self) -> None:
        """Resolve async capabilities and revalidate.

        Idempotent — calling twice is a no-op.
        """
        if self._prepared:
            return

        if self.skills_registry is not None:
            loaded = await self.skills_registry.load(self.skills_registry.skills)
            self._loaded_skills = loaded
            # Build the global read_skill_resource tool over the loaded
            # catalog. Only meaningful when at least one skill loaded.
            if loaded:
                self._read_resource_tool = self.skills_registry.make_read_resource_tool(
                    loaded
                )
            # Revalidate with skill tools merged in.
            self._validate_unique_tool_names(self._all_tools())

        self._prepared = True

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise CapabilityError.not_prepared()

    # -------- VALIDATION ---------------------------------------------
    def _validate_priority_tools_exist(self) -> None:
        """Every priority_tools entry must match a real tool name.

        Only checks against sync-visible tools. Priority tools can't
        point at skill tools anyway — priority is declared by the
        agent author, and skill tools are loaded dynamically.
        """
        tool_names = {tool.name for tool in self._all_tools_sync()}
        missing = [name for name in self.priority_tools if name not in tool_names]
        if missing:
            raise CapabilityError.invalid_priority_tool(
                missing=missing, tool_names=tool_names
            )

    @staticmethod
    def _validate_unique_tool_names(tools: t.Sequence[CoreTool]) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                duplicates.add(tool.name)
            seen.add(tool.name)
        if duplicates:
            raise CapabilityError.duplicate_tool_names(duplicates=duplicates)

    # -------- TOOL COLLECTION ----------------------------------------
    def _all_tools_sync(self) -> list[CoreTool]:
        """All tools available before prepare() — explicit + memory + knowledge + routines + context."""
        merged: list[CoreTool] = list(self.toolset)
        merged.extend(self.memory.tools if self.memory else [])
        merged.extend(self.logbook.tools if self.logbook else [])
        merged.extend(self.routines.tools if self.routines else [])
        merged.extend(k for registry in self.knowledge for k in registry.tools)
        return merged

    def _all_tools(self) -> list[CoreTool]:
        """All tools including skills (only valid post-prepare)."""
        return self._all_tools_sync() + self.skill_tools

    @property
    def all_tools(self) -> list[CoreTool]:
        """Complete set of tools exposed to the LLM. Requires prepare()."""
        self._ensure_prepared()
        return self._all_tools()

    # -------- LOOKUPS ------------------------------------------------
    def find_tool(self, name: str) -> CoreTool | None:
        """Lookup a tool by name across all sources. Returns None if not found."""
        return next((t for t in self.all_tools if t.name == name), None)

    def get_tool(self, name: str) -> CoreTool:
        """Lookup a tool by name. Raises if not found."""
        tool = self.find_tool(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool

    # -------- CAPABILITY-DERIVED TOOLS -------------------------------
    @property
    def skill_tools(self) -> list[CoreTool]:
        """Tools exposed by loaded skills.

        Includes both the user-defined tools from each skill's
        ``scripts/`` and the global ``read_skill_resource`` tool
        provided by the registry. Empty before ``prepare()``.
        """
        tools: list[CoreTool] = []
        for skill in self._loaded_skills:
            tools.extend(skill.tools)
        if self._read_resource_tool is not None:
            tools.append(self._read_resource_tool)
        return tools

    # -------- QUERIES ------------------------------------------------
    @property
    def has_tools(self) -> bool:
        return bool(self.toolset)

    @property
    def has_memory(self) -> bool:
        return self.memory is not None

    @property
    def has_knowledge(self) -> bool:
        return bool(self.knowledge)

    @property
    def has_routines(self) -> bool:
        return self.routines is not None

    @property
    def has_skills(self) -> bool:
        return self.skills_registry is not None

    @property
    def has_logbook(self) -> bool:
        return self.logbook is not None

    @property
    def loaded_skills(self) -> list[Skill]:
        """Resolved Skill objects. Empty before prepare()."""
        return list(self._loaded_skills)

    def __repr__(self) -> str:
        return (
            f"CapabilityRegistry("
            f"memory={self.has_memory}, "
            f"tools={len(self.toolset)}, "
            f"knowledge={len(self.knowledge)}, "
            f"routines={self.has_routines}, "
            f"skills={len(self._loaded_skills) if self._prepared else '?'}, "
            f"context={self.has_logbook}, "
            f"priority_tools={len(self.priority_tools)}, "
            f"prepared={self._prepared})"
        )
