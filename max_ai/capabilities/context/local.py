"""
Filesystem-backed conversation-context registry.

Stores per-session summaries for a single user in one JSON file under
``{base_path}/context/{user_id}.json``. Each session entry holds the
summary, an optional vector for future semantic search, a timestamp,
and free-form metadata.

This registry is strictly read-only from the agent's perspective.
Summarization, vectorization, and writing to disk are done OUTSIDE the
agent — typically by a job that runs after a session ends. The agent
only consumes what's already there.

Not concurrency-safe: simultaneous external writers can clobber each
other. Intended for local development and tests.
"""

from __future__ import annotations

import json
import random
import typing as t
from datetime import datetime
from pathlib import Path

from ...base.context import CoreContextRegistry, ContextToolMode, ContextBlock


class LocalContextRegistry(CoreContextRegistry):
    """Filesystem-backed implementation of ``CoreContextRegistry``.

    Layout::

        {base_path}/
            context/
                {user_id}.json        # {session_id: {summary, vector, timestamp, metadata}}

    File shape::

        {
            "session_001": {
                "summary": "We discussed spaceship designs...",
                "vector": [],
                "timestamp": "2026-04-20T15:00:00Z",
                "metadata": {}
            },
            "session_002": { ... }
        }

    The ``vector`` field is reserved for future semantic search but
    ignored by this backend — ``search()`` uses token overlap as a
    placeholder scorer until real embeddings are wired in.
    """

    # Each session entry must have these fields. Missing fields → file
    # treated as corrupt, fail loud.
    _REQUIRED_SESSION_FIELDS: t.ClassVar[set[str]] = {
        "summary",
        "vector",
        "timestamp",
        "metadata",
    }

    def __init__(
        self,
        user_id: str,
        session_id: str,
        base_path: str | Path,
        tool_mode: ContextToolMode = ContextToolMode.READ_ONLY,
    ) -> None:
        super().__init__(
            user_id=user_id, session_id=session_id, tool_mode=tool_mode
        )
        self.base_path: Path = Path(base_path).expanduser().resolve()

    # -------- PATH HELPERS -----------------------------------------------------------
    @property
    def _context_dir(self) -> Path:
        return self.base_path / "context"

    @property
    def _user_file(self) -> Path:
        return self._context_dir / f"{self.user_id}.json"

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Ensure the context directory exists. The user file itself
        is created externally — readers tolerate its absence."""
        self._context_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        return None

    # -------- READ FOR PROMPT INJECTION -----------------------------------------------------------
    async def get_current_session_summary(self) -> str | None:
        """Return the stored summary for the current session, if any.

        Returns ``None`` when:
          - the user has no context file yet,
          - the current ``session_id`` is not in the file, or
          - the entry exists but its summary is empty / whitespace.
        """
        await self._ensure_connected()
        store = self._read_all()
        entry = store.get(self.session_id)
        if entry is None:
            return None
        summary = entry.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return summary

    # -------- READ FOR TOOL -----------------------------------------------------------
    async def search(
        self, query: str, limit: int = 5
    ) -> list[ContextBlock]:
        """Search across this user's stored sessions.

        Uses token overlap as a placeholder scoring function. Sessions
        with zero overlap are dropped; the rest are sorted by score
        descending and the top ``limit`` are returned.
        """
        await self._ensure_connected()
        if not isinstance(query, str) or not query.strip():
            return []

        store = self._read_all()
        if not store:
            return []

        scored: list[tuple[float, ContextBlock]] = []
        for session_id, entry in store.items():
            summary = entry.get("summary", "") or ""
            score = self._fake_score(query, summary)
            if score <= 0:
                continue
            scored.append((score, self._entry_to_block(session_id, entry, score)))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [block for _, block in scored[:limit]]

    # -------- INTERNALS -----------------------------------------------------------
    @staticmethod
    def _fake_score(query: str, content: str) -> float:
        """Token-overlap score with a tiny random tiebreaker.

        Returns 0.0 when there's no overlap. Otherwise returns
        ``overlap / len(query_tokens)`` plus a small random nudge so
        ties between sessions don't always sort the same way.
        """
        query_tokens = {t for t in query.lower().split() if t}
        if not query_tokens:
            return 0.0
        content_tokens = {t for t in content.lower().split() if t}
        overlap = len(query_tokens & content_tokens)
        if overlap == 0:
            return 0.0
        base = overlap / len(query_tokens)
        return base + random.uniform(0, 0.001)

    @staticmethod
    def _entry_to_block(
        session_id: str, entry: dict[str, t.Any], score: float
    ) -> ContextBlock:
        """Build a ``ContextBlock`` from a stored session entry."""
        timestamp_raw = entry.get("timestamp")
        if isinstance(timestamp_raw, str):
            timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        elif isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        else:
            raise ValueError(
                f"Session {session_id!r}: timestamp must be an ISO 8601 "
                f"string or datetime, got {type(timestamp_raw).__name__}"
            )

        return ContextBlock(
            session_id=session_id,
            timestamp=timestamp,
            content=entry.get("summary", "") or "",
            score=score,
            metadata=entry.get("metadata") or {},
        )

    def _read_all(self) -> dict[str, dict[str, t.Any]]:
        """Load the user's full context file from disk.

        Returns ``{}`` when the file doesn't exist yet — first read
        for a new user is not an error. A corrupt file or a session
        entry missing required fields fails loud.
        """
        path = self._user_file
        if not path.is_file():
            return {}

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Context file {path} is not valid JSON: {e}"
            ) from e

        if not isinstance(raw, dict):
            raise ValueError(
                f"Context file {path} must contain a JSON object at the "
                f"top level, got {type(raw).__name__}"
            )

        for session_id, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Context file {path}: session {session_id!r} must "
                    f"be a JSON object, got {type(entry).__name__}"
                )
            missing = self._REQUIRED_SESSION_FIELDS - entry.keys()
            if missing:
                raise ValueError(
                    f"Context file {path}: session {session_id!r} is "
                    f"missing required fields: {sorted(missing)}"
                )
        return raw
