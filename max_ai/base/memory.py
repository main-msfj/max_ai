"""
Core contract for agent memory backends.

A memory is a connection to a persistence layer (Redis, Postgres, in-memory,
whatever) that stores facts about a user across sessions. It exposes two
surfaces to the agent:

  1. ``get_context()`` → read-only access used by the framework to inject
     relevant memories into the prompt at the start of a turn.
  2. ``as_tools()``    → the CRUD tools the LLM can call to read, create,
     update, or delete memories during the turn. What tools are exposed
     is controlled by ``tool_mode``.

Subclasses implement the storage-level methods (list_facts, update_fact,
delete_fact). The base class handles tool generation.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from .tools import CoreTool
from .component_config import ComponentBase

from ..core import MemoryBlock
from ..loggers import ScopedLogger
from ..types.tools import ToolApprovalMode

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="CoreMemoryRegistry")


class MemoryToolMode(str, Enum):
    """Controls which memory tools are exposed to the LLM.

    - ``NONE``      : No tools. Memory is read into the prompt via
                      ``get_context()`` but the LLM cannot modify it.
                      Use when writes happen outside the agent (batch
                      jobs, extractors, manual UIs).
    - ``READ_ONLY`` : Only ``list_memories`` is exposed. The LLM can
                      query the full memory but cannot modify it.
    - ``FULL``      : ``list_memories`` + ``update_memory`` +
                      ``delete_memory``. The LLM has full CRUD agency.
    """

    NONE = "none"
    READ_ONLY = "read_only"
    FULL = "full"


class CoreMemoryRegistry(ComponentBase[BaseModel], ABC):
    """Abstract base class for agent memory backends."""

    def __init__(
        self,
        user_id: str,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
    ) -> None:
        self._connected: bool = False
        self.user_id: str = self.require_type(user_id, str, "user_id")
        self.tool_mode: MemoryToolMode = self.require_type(
            tool_mode, MemoryToolMode, "tool_mode"
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
                    log.info(msg=f"Connecting memory backend for user {self.user_id}")
                    await self.connect()
                    self._connected = True

    async def __aenter__(self) -> t.Self:
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
        if self._connected:
            await self.disconnect()
            self._connected = False

    # -------- OPERATION -----------------------------------------------------------
    @abstractmethod
    async def get_context(self) -> list[MemoryBlock]:
        """Return relevant memory blocks for prompt injection."""
        ...

    # -------- CRUD -----------------------------------------------------------
    @abstractmethod
    async def list_facts(self) -> list[MemoryBlock]:
        """List all facts stored for this user."""
        ...

    @abstractmethod
    async def update_fact(self, key: str, value: str) -> None:
        """Create or update a fact in memory."""
        ...

    @abstractmethod
    async def delete_fact(self, key: str) -> None:
        """Delete a fact from memory."""
        ...

    # -------- READ EXPOSOR -----------------------------------------------------------
    def as_tools(self) -> list[CoreTool]:
        """
        Generate the tools the LLM can call to interact with this memory.

        The set of tools depends on ``self.tool_mode``:

        - ``NONE``      → []
        - ``READ_ONLY`` → [list_memories]
        - ``FULL``      → [list_memories, update_memory, delete_memory]

        Subclasses can override this to add custom tools (e.g. bulk-import,
        search-by-tag) or to expose a completely different set.
        """
        if self.tool_mode == MemoryToolMode.NONE:
            return []
        if self.tool_mode == MemoryToolMode.READ_ONLY:
            return [self._build_list_tool()]
        return [
            self._build_list_tool(),
            self._build_update_tool(),
            self._build_delete_tool(),
        ]
    # -------- TOOL FACTORIES -----------------------------------------------------------
    def _build_list_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool
        """Tool: list every fact currently stored for the user."""

        async def list_memories() -> list[dict[str, t.Any]]:
            """List every fact currently stored about the user."""
            await self._ensure_connected()
            facts = await self.list_facts()
            return [f.model_dump() for f in facts]

        return FunctionAsTool(
            list_memories,
            name="list_memories",
            description=(
                "List every fact currently stored about the user across "
                "past sessions."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_update_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool
        """Tool: create or update a durable fact about the user."""

        async def update_memory(key: str, value: str) -> str:
            """Create or update a durable fact about the user.

            Args:
                key: Short identifier for the fact (e.g. "location",
                    "preferred_language", "timezone").
                value: The fact content to store or overwrite.
            """
            await self._ensure_connected()
            await self.update_fact(key, value)
            return f"Memory saved: {key}"

        return FunctionAsTool(
            update_memory,
            name="update_memory",
            description=(
                "Store or overwrite a durable fact about the user. Use "
                "this when the user shares information worth remembering "
                "across sessions (preferences, personal context, recurring "
                "goals). Do NOT store transient or conversational details."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )

    def _build_delete_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool
        """Tool: remove a stored fact about the user."""

        async def delete_memory(key: str) -> str:
            """Delete a previously stored fact about the user.

            Args:
                key: Identifier of the fact to remove.
            """
            await self._ensure_connected()
            await self.delete_fact(key)
            return f"Memory deleted: {key}"

        return FunctionAsTool(
            delete_memory,
            name="delete_memory",
            description=(
                "Delete a previously stored fact about the user. Use when "
                "the user explicitly asks to forget something, or when "
                "newer information has made a stored fact obsolete."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )
