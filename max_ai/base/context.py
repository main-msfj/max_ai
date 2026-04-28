"""
Core contract for agent conversation context.

A context registry stores summaries of past sessions for a single user.
The agent uses it for two things:

  1. ``get_current_session_summary()`` → read the stored summary for the
     active session (if any), for injection into the system prompt as
     continuity context.
  2. ``as_tools()`` → expose ``search_context``, a semantic-search tool
     the LLM can call to recall past conversations across sessions
     belonging to this user.

Summarization, persistence, and lifecycle of stored content live OUTSIDE
the agent. A separate process — typically run after a session ends — is
responsible for generating summaries and writing them to the storage
backend. This registry is strictly read-only from the agent's point of
view.

Subclasses implement ``get_current_session_summary`` and ``search``. The
base class handles tool generation.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel

from .component_config import ComponentBase
from .tools import CoreTool
from ..loggers import ScopedLogger
from ..tools.function_as_tool import FunctionAsTool
from ..types.tools import ToolApprovalMode
from ..core.blocks import ContextBlock

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="CoreContextRegistry")


class ContextToolMode(str, Enum):
    """Controls whether the context-search tool is exposed to the LLM.

    - ``NONE``      : No tool. The current-session summary may still be
                      injected into the system prompt, but the LLM
                      cannot search past sessions.
    - ``READ_ONLY`` : ``search_context`` is exposed. The LLM can recall
                      past conversations on demand.
    """

    NONE = "none"
    READ_ONLY = "read_only"


class CoreContextRegistry(ComponentBase[BaseModel], ABC):
    """Abstract base class for read-only conversation context.

    A registry instance is scoped to a single ``(user_id, session_id)``
    pair. The ``user_id`` defines which past sessions are searchable;
    the ``session_id`` identifies which session's summary to inject
    into the system prompt.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        tool_mode: ContextToolMode = ContextToolMode.READ_ONLY,
    ) -> None:
        self.user_id: str = self.require_type(user_id, str, "user_id")
        self.session_id: str = self.require_type(session_id, str, "session_id")
        self.tool_mode: ContextToolMode = self.require_type(
            tool_mode, ContextToolMode, "tool_mode"
        )
        self._connected: bool = False
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
                    log.info(
                        msg=(
                            f"Connecting context registry for "
                            f"user={self.user_id!r} session={self.session_id!r}"
                        )
                    )
                    await self.connect()
                    self._connected = True

    async def __aenter__(self) -> t.Self:
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
        if self._connected:
            await self.disconnect()
            self._connected = False

    # -------- READ FOR PROMPT INJECTION -----------------------------------------------------------
    @abstractmethod
    async def get_current_session_summary(self) -> str | None:
        """Return the stored summary for the current session, if any.

        Returns ``None`` for fresh sessions or sessions that have not
        been summarized yet. The framework injects the result into the
        system prompt via ``ContextLayer`` during ``agent.prepare()``.
        """
        ...

    # -------- READ FOR TOOL -----------------------------------------------------------
    @abstractmethod
    async def search(
        self, query: str, limit: int = 5
    ) -> list[ContextBlock]:
        """Semantic search across this user's stored sessions.

        The storage layer filters by ``self.user_id`` implicitly — the
        LLM never sees or specifies a user identifier. Returns up to
        ``limit`` blocks ranked by relevance.
        """
        ...

    # -------- TOOL EXPOSURE -----------------------------------------------------------
    def as_tools(self) -> list[CoreTool]:
        """Generate the tools the LLM can call against this registry.

        Depends on ``self.tool_mode``:
          - ``NONE``      → []
          - ``READ_ONLY`` → [search_context]
        """
        if self.tool_mode == ContextToolMode.NONE:
            return []
        return [self._build_search_tool()]

    # -------- TOOL FACTORIES -----------------------------------------------------------
    def _build_search_tool(self) -> CoreTool:
        """Tool: semantic search over past sessions with this user."""

        async def search_context(
            query: str, limit: int = 5
        ) -> list[dict[str, t.Any]]:
            """Search past conversations with this user.

            Use this when the user references something from a previous
            conversation that you don't currently see. Phrase the query
            as a description of what they're asking you to recall.

            Args:
                query: Natural-language description of what to recall.
                limit: Maximum number of fragments to return.
            """
            await self._ensure_connected()
            blocks = await self.search(query, limit=limit)
            return [b.model_dump() for b in blocks]

        return FunctionAsTool(
            search_context,
            name="search_context",
            description=(
                "Search past conversations with this user for relevant "
                "context. Call this when the user references something "
                "from a previous session that isn't visible in the "
                "current context."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
