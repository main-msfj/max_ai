"""
Capability Registry — normalizes and validates the agent's runtime
components (tools, memory, knowledge, routines, skills, context) before
the agent is used.

Responsibilities:
  - Coerce heterogeneous inputs to their canonical types.
  - Validate internal coherence (e.g. priority_tools reference real tools).
  - Resolve async capability metadata (skills) via ``prepare()``.
  - Provide type-safe lookup APIs for the agent at runtime.

Out of scope:
  - Provider-specific formatting (that lives in each client's
    build_tool_schema / adapt_schema_for_provider).
  - Prompt variable assembly (that lives in PromptVariablesBuilder).
"""

from __future__ import annotations

import logging
import typing as t
from pydantic import BaseModel

from ..loggers import ScopedLogger
from ..errors.capabilities import CapabilityError
from ..base.capability import CoreAgentCapabilities
from ..base.context import CoreLogBookRegistry
from ..base.knowledge import CoreKnowledgeRegistry
from ..base.memory import CoreMemoryRegistry
from ..base.routines import CoreRoutineRegistry
from ..base.skills import CoreSkillRegistry
from ..base.tools import CoreTool
from ..base.workspace import WorkSpaceRegistry
from ..tools.bash import BashTool
from ..tools.function_as_tool import FunctionAsTool
from ..tools.workspace import WorkspaceTool

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, prefix="[AgentCapabilities]")
CapabilitiesT: t.TypeAlias = CoreAgentCapabilities[BaseModel]

if t.TYPE_CHECKING:
    from ..core.blocks import SkillBlock
    from ..types.workspace import WorkspaceDirectory


class AgentCapabilities:
    """
    Central registry of the agent's capabilities.

    Built in two phases:
      1. ``__init__`` — sync. Normalizes inputs, validates what it can
         see right away (explicit tools, priority_tools, name uniqueness
         across tools/memory/knowledge/routines/context).
      2. ``prepare()`` — async. Loads lightweight skill metadata via
         the skills registry. Skill registries do not expose tools;
         skill execution is handled by a separate runtime tool.

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
        capabilities: t.Sequence[CapabilitiesT | t.Sequence[CapabilitiesT]]
        | None = None,
        *,
        priority_tools: t.Sequence[str] | None = None,
        toolset: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
    ) -> None:
        """
        Args:
            capabilities: Optional flat or partially nested sequence of
                CoreAgentCapabilities.
        """
        self.capabilities: list[CapabilitiesT] = []
        self.memory: CoreMemoryRegistry | None = None
        self.skills: CoreSkillRegistry | None = None
        self.logbook: CoreLogBookRegistry | None = None
        self.routines: CoreRoutineRegistry | None = None
        self.workspace: WorkSpaceRegistry | None = None
        self.priority_tools: list[str] = list(priority_tools or [])
        self.knowledge: list[CoreKnowledgeRegistry] = []
        self.toolset: list[CoreTool] = self._normalize_tools(toolset or [])
        self._skill_blocks: list[SkillBlock] = []  # type: ignore[name-defined]
        self._prepared: bool = False

        for cap in self._flatten_capabilities(capabilities or []):
            self.capabilities.append(cap)
            self._register_capability(cap)

        self._validate_priority_tools_exist()
        self._validate_unique_tool_names(self._all_tools_sync())

    @staticmethod
    def _flatten_capabilities(
        capabilities: t.Iterable[CapabilitiesT | t.Sequence[CapabilitiesT]],
    ) -> list[CapabilitiesT]:
        flat: list[CapabilitiesT] = []
        for cap in capabilities:
            if isinstance(cap, CoreAgentCapabilities):
                flat.append(cap)
                continue
            for item in cap:
                if not isinstance(item, CoreAgentCapabilities):
                    raise TypeError(
                        "capabilities must contain CoreAgentCapabilities instances"
                    )
                flat.append(item)
        return flat

    def _register_capability(self, cap: CapabilitiesT) -> None:
        if isinstance(cap, CoreMemoryRegistry) and self.memory is None:
            self.memory = cap
        elif isinstance(cap, CoreSkillRegistry) and self.skills is None:
            self.skills = cap
        elif isinstance(cap, CoreLogBookRegistry) and self.logbook is None:
            self.logbook = cap
        elif isinstance(cap, CoreRoutineRegistry) and self.routines is None:
            self.routines = cap
        elif isinstance(cap, CoreKnowledgeRegistry) and cap not in self.knowledge:
            self.knowledge.append(cap)
        elif isinstance(cap, WorkSpaceRegistry) and self.workspace is None:
            self.workspace = cap

    @staticmethod
    def _normalize_tools(
        tools: t.Sequence[CoreTool | t.Callable[..., t.Any]],
    ) -> list[CoreTool]:
        normalized: list[CoreTool] = []
        for tool in tools:
            if isinstance(tool, CoreTool):
                normalized.append(tool)
            elif callable(tool):
                normalized.append(FunctionAsTool(tool))
            else:
                raise CapabilityError.invalid_tool_entry(tool)
        return normalized

    def get_native_tools(self) -> list[CoreTool]:
        tools: list[CoreTool] = [WorkspaceTool()]
        if self.skills is not None:
            tools.append(BashTool())
        return tools

    def collect_tools(self) -> list[CoreTool]:
        """
        Return all tools provided by all capabilities.
        """
        return self._all_tools_sync()

    async def prepare(self) -> None:
        """
        Resolve async capabilities and skill blocks.
        Idempotent — calling twice is a no-op.
        """
        if getattr(self, "_prepared", False):
            return

        if self.skills is not None:
            self._skill_blocks = await self.skills.get_skills()

        self._validate_unique_tool_names(self._all_tools_sync())
        self._prepared = True

    def materialize_runtime(self, directory: "WorkspaceDirectory") -> None:
        """Materialize per-user runtime assets for registered capabilities."""
        if self.skills is not None:
            self.skills.materialize(directory)

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise CapabilityError.not_prepared()

    def _validate_priority_tools_exist(self) -> None:
        tool_names = {tool.name for tool in self._all_tools_sync()}
        missing = [name for name in self.priority_tools if name not in tool_names]
        if missing:
            raise CapabilityError.invalid_priority_tool(missing, tool_names)

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

    def _all_tools_sync(self) -> list[CoreTool]:
        merged: list[CoreTool] = []
        merged.extend(self.toolset)
        merged.extend(self.memory.tools if self.memory else [])
        merged.extend(self.logbook.tools if self.logbook else [])
        merged.extend(self.routines.tools if self.routines else [])
        merged.extend(tool for registry in self.knowledge for tool in registry.tools)
        merged.extend(self.get_native_tools())
        return merged

    @property
    def all_tools(self) -> list[CoreTool]:
        self._ensure_prepared()
        return self._all_tools_sync()

    def find_tool(self, name: str) -> CoreTool | None:
        return next((tool for tool in self.all_tools if tool.name == name), None)

    def get_tool(self, name: str) -> CoreTool:
        tool = self.find_tool(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool

    @property
    def has_tools(self) -> bool:
        return bool(self.toolset)

    @property
    def has_capability_tools(self) -> bool:
        """Return True when a registered capability exposes tools."""
        return any(bool(cap.tools) for cap in self.capabilities)

    @property
    def requires_workspace(self) -> bool:
        """Return True when runtime files are needed."""
        return self.has_tools or self.has_skills

    @property
    def requires_sandbox_executor(self) -> bool:
        """Return True when local execution should not be allowed."""
        return self.has_skills

    @property
    def has_memory(self) -> bool:
        """Return True if a memory capability is present."""
        return self.memory is not None

    @property
    def has_knowledge(self) -> bool:
        """Return True if at least one knowledge capability is present."""
        return bool(self.knowledge)

    @property
    def has_routines(self) -> bool:
        """Return True if a routines capability is present."""
        return self.routines is not None

    @property
    def has_skills(self) -> bool:
        """Return True if a skills capability is present."""
        return self.skills is not None

    @property
    def has_logbook(self) -> bool:
        """Return True if a logbook capability is present."""
        return self.logbook is not None

    @property
    def skill_blocks(self) -> list[SkillBlock]:
        """
        Return all SkillBlock entries collected from the skill registry.
        """
        return list(self._skill_blocks)

    @property
    def loaded_skill_blocks(self) -> list[SkillBlock]:
        """Lightweight SkillBlock entries for the prompt layer."""
        return list(self._skill_blocks)

    def __repr__(self) -> str:
        """
        Human-readable representation of the agent's capabilities,
        including skills and number of tools.
        """
        return (
            f"AgentCapabilities("
            f"memory={self.has_memory}, "
            f"tools={len(self.toolset)}, "
            f"knowledge={len(self.knowledge)}, "
            f"routines={self.has_routines}, "
            f"skills={len(self._skill_blocks) if self._prepared else '?'}, "
            f"context={self.has_logbook}, "
            f"priority_tools={len(self.priority_tools)}, "
            f"prepared={self._prepared})"
        )
