"""
Core contract for agent routine registries.

A routine is a pre-defined procedure expressed as text (how to write an
article, how to draft an email, etc.). Routines are read-only — the LLM
never creates or modifies them, only discovers and loads them.

A ``CoreRoutineRegistryRegistry`` is a connection to a storage layer (DB, files,
in-memory) that holds a catalog of routines. It exposes two surfaces to
the agent:

  1. ``get_catalog()`` → read-only listing used by the framework to
     inject a lightweight catalog (name + description) into the prompt
     when the agent wants the LLM to know what's available.
  2. ``as_tools()``    → the discovery/load tools the LLM can call.
     What tools are exposed is controlled by ``tool_mode``.

Subclasses implement the storage-level methods (search, fetch, list).
The base class handles tool generation.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from .component_config import ComponentBase
from .tools import CoreTool
from ..types.tools import ToolApprovalMode
from ..core.blocks import RoutineBlocks
from ..loggers import ScopedLogger

if t.TYPE_CHECKING:
    from ..types.routines import RoutineSummary

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="CoreRoutineRegistryRegistry")


class RoutineToolMode(str, Enum):
    """Controls which routine tools are exposed to the LLM.

    - ``NONE``      : No tools. Routines are surfaced through other
                      channels (e.g. inline catalog in the prompt, or
                      ``@routine`` triggers resolved by a middleware).
    - ``LOAD_ONLY`` : Only ``get_routine`` is exposed. Use when the
                      catalog is already visible to the LLM in the
                      prompt and it only needs to fetch the full body
                      by name.
    - ``FULL``      : ``search_routines`` + ``get_routine``. The LLM
                      discovers routines semantically and loads the
                      best match. Recommended for catalogs of 20+
                      routines that won't fit inline.
    """

    NONE = "none"
    LOAD_ONLY = "load_only"
    FULL = "full"


class CoreRoutineRegistry(ComponentBase[BaseModel], ABC):
    """Abstract base class for agent routine registries."""

    def __init__(
        self,
        tool_mode: RoutineToolMode = RoutineToolMode.FULL,
    ) -> None:
        self._connected: bool = False
        self.tool_mode: RoutineToolMode = self.require_type(
            tool_mode, RoutineToolMode, "tool_mode"
        )
        self._connect_lock: asyncio.Lock = asyncio.Lock()

    # -------- LIFECYCLE -----------------------------------------------------------
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    async def _ensure_connected(self) -> None:
        if not self._connected:
            async with self._connect_lock:
                if not self._connected:
                    log.info(msg="Connecting routine registry")
                    await self.connect()
                    self._connected = True

    async def __aenter__(self) -> t.Self:
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
        if self._connected:
            await self.disconnect()
            self._connected = False

    # -------- AGENT BOUNDARY -----------------------------------------------------------
    @abstractmethod
    async def get_catalog(self) -> list["RoutineSummary"]:
        """Return a lightweight catalog (name + description) for prompt
        injection. Used by the framework when the LLM should be aware of
        available routines without loading their full content."""
        ...

    # -------- READ OPERATIONS -----------------------------------------------------------
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list["RoutineSummary"]:
        """Semantic search over the routine catalog.

        Returns up to ``limit`` candidate routines ranked by relevance
        to ``query``. Each result is a lightweight summary
        (name + description), NOT the full instructions.
        """
        ...

    @abstractmethod
    async def fetch(self, name: str) -> RoutineBlocks:
        """Load the full routine by name, including instructions."""
        ...

    # -------- TOOL EXPOSURE -----------------------------------------------------------
    def as_tools(self) -> list[CoreTool]:
        """
        Generate the tools the LLM can call to discover and load routines.

        The set of tools depends on ``self.tool_mode``:

        - ``NONE``      → []
        - ``LOAD_ONLY`` → [get_routine]
        - ``FULL``      → [search_routines, get_routine]

        Subclasses can override this to add custom tools or change the
        exposed set.
        """
        if self.tool_mode == RoutineToolMode.NONE:
            return []
        if self.tool_mode == RoutineToolMode.LOAD_ONLY:
            return [self._build_get_tool()]
        return [
            self._build_search_tool(),
            self._build_get_tool(),
        ]

    @property
    def tools(self) -> list[CoreTool]:
        """Public capability surface for tool aggregation."""
        return self.as_tools()

    # -------- TOOL FACTORIES -----------------------------------------------------------
    def _build_search_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool
        """Tool: semantic search over the routine catalog."""

        async def search_routines(query: str, limit: int = 5) -> list[dict[str, t.Any]]:
            """Search the routine catalog for procedures matching a task.

            Returns a short list of candidate routines with name and
            description. Use this when the user's request looks like
            a recurring task type (writing, drafting, reviewing, etc.)
            and you want to know which routines apply.

            Args:
                query: Short natural-language description of the task.
                limit: Maximum number of candidates to return.
            """
            await self._ensure_connected()
            results = await self.search(query, limit=limit)
            return [r.model_dump() for r in results]

        return FunctionAsTool(
            search_routines,
            name="search_routines",
            description=(
                "Semantic search over the routine catalog. Returns "
                "candidate procedures that match a given task "
                "description. Call this before get_routine to pick the "
                "right one."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_get_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool
        """Tool: load the full routine instructions by name."""

        async def get_routine(name: str) -> str:
            """Load the full instructions for a specific routine.

            The returned text is authoritative guidance for the task —
            follow it as written, and it takes precedence over your
            general defaults for that specific task.

            Args:
                name: Unique identifier of the routine to load
                    (e.g. "article", "email_reply", "code_review").
            """
            await self._ensure_connected()
            routine = await self.fetch(name)
            return routine.instructions

        return FunctionAsTool(
            get_routine,
            name="get_routine",
            description=(
                "Load the full instructions for a specific routine by "
                "name. Call this after search_routines has identified "
                "the best match, or when the user explicitly names a "
                "routine to apply."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
