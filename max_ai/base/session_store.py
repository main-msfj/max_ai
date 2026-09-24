"""Contract for saving and resuming conversations (``RunContext``).

The Agent never touches a store: the host (a server, the CLI) does
``load → agent.run → save``. Sessions are keyed by ``(user_id,
session_id)`` so one user can never load another user's session.
"""

from __future__ import annotations

import re
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..core.messages import HARNESS_SOURCE, UserMessage
from ..core.model.session import SessionInfo
from ..types.run_context import RunContext
from .component import CoreLifecycleComponent

# Ids become file names and database keys: no traversal, no escaping.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_id(kind: str, value: str) -> str:
    """
    Validate an identifier before using it as a storage key.

    Parameters
    ----------
    kind : str
        Identifier category included in validation errors.
    value : str
        Value to validate.

    Returns
    -------
    str
        The resulting text value.
    """
    if not isinstance(value, str) or not _SAFE_ID.match(value):
        raise ValueError(f"Invalid {kind} {value!r}: use 1-128 of [A-Za-z0-9_-]")
    return value


def session_info(ctx: RunContext) -> SessionInfo:
    """Listing entry for ``ctx``; the title is its first real user message."""
    first = next(
        (m.text() for m in ctx.messages
         if isinstance(m, UserMessage) and m.source != HARNESS_SOURCE and m.text().strip()),
        "",
    )
    title = " ".join(first.split())
    return SessionInfo(
        user_id=ctx.user_id,
        session_id=ctx.session_id or "",
        title=title if len(title) <= 80 else title[:79] + "…",
        message_count=len(ctx.messages),
        compactions=ctx.compaction.compactions,
    )


class CoreSessionStore(CoreLifecycleComponent[BaseModel], ABC):
    """Saves whole ``RunContext``s (messages, compaction summary, plan and
    paused tool calls), so a conversation resumes exactly where it was.

    Public methods validate ids and connect lazily; backends implement the
    ``_load`` / ``_save`` / ``_list`` / ``_delete`` hooks.
    """

    component_type = "session_store"

    async def load(self, user_id: str, session_id: str) -> RunContext | None:
        """The saved context, or ``None`` when that session doesn't exist."""
        validate_id("user_id", user_id)
        validate_id("session_id", session_id)
        await self._ensure_connected()
        ctx = await self._load(user_id, session_id)
        if ctx is not None and (ctx.user_id, ctx.session_id) != (user_id, session_id):
            raise ValueError(f"Stored session {session_id!r} does not belong to {user_id!r}")
        return ctx

    async def save(self, ctx: RunContext) -> SessionInfo:
        """Create or replace the session ``(ctx.user_id, ctx.session_id)``."""
        validate_id("user_id", ctx.user_id)
        validate_id("session_id", ctx.session_id or "")
        await self._ensure_connected()
        info = session_info(ctx)
        await self._save(ctx, info)
        return info

    async def list_sessions(self, user_id: str, limit: int = 50) -> list[SessionInfo]:
        """The user's sessions, most recently updated first."""
        validate_id("user_id", user_id)
        await self._ensure_connected()
        sessions = await self._list(user_id)
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)[:limit]

    async def delete(self, user_id: str, session_id: str) -> bool:
        """Remove a session. ``False`` when it didn't exist."""
        validate_id("user_id", user_id)
        validate_id("session_id", session_id)
        await self._ensure_connected()
        return await self._delete(user_id, session_id)

    # -------- BACKEND HOOKS -----------------------------------------------------------
    @abstractmethod
    async def _load(self, user_id: str, session_id: str) -> RunContext | None:
        """
        Load one saved run context from the backend.

        Parameters
        ----------
        user_id : str
            Identifier for the user scope.
        session_id : str
            Identifier for the current session.

        Returns
        -------
        RunContext | None
            The saved run context, or None when the session is absent.
        """
        ...

    @abstractmethod
    async def _save(self, ctx: RunContext, info: SessionInfo) -> None:
        """
        Persist a run context and its listing metadata.

        Parameters
        ----------
        ctx : RunContext
            Current run context.
        info : SessionInfo
            Metadata describing the saved session.
        """
        ...

    @abstractmethod
    async def _list(self, user_id: str) -> t.Iterable[SessionInfo]:
        """
        Return the saved session metadata for one user.

        Parameters
        ----------
        user_id : str
            Identifier for the user scope.

        Returns
        -------
        t.Iterable[SessionInfo]
            The iterable of saved session metadata.
        """
        ...

    @abstractmethod
    async def _delete(self, user_id: str, session_id: str) -> bool:
        """
        Delete one saved session from the backend.

        Parameters
        ----------
        user_id : str
            Identifier for the user scope.
        session_id : str
            Identifier for the current session.

        Returns
        -------
        bool
            Whether the operation succeeded.
        """
        ...
