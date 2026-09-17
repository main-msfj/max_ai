"""
Core contract for agent memory backends (corrected).

One registry == one user, fixed at construction. No scope levels — tenant
separation, if ever needed, is "which store does this instance point at",
a deployment choice, not a contract concern.

What a concrete backend implements (storage surface only):
    connect / disconnect            — connection lifecycle
    _read_all()                     — every record for this user
    _write_many(records)            — batch upsert
    _delete_many(keys)              — batch delete
    recall(query)  [OPTIONAL]       — override to push ranking to a native
                                      index; the default ranks in Python.

Everything else — relevance-aware injection, merge/dedup, batch convenience,
and the CRUD tools exposed to the LLM — is shared here so every backend
behaves identically.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from .tools import CoreTool
from .capability import CoreAgentCapabilities
from ..types.tools import ToolApprovalMode

# Two embedders, injected separately. `EmbedOne` embeds a single text
# (used to embed the query at recall time and to compare pairs during
# merge). `EmbedMany` embeds a batch in one model call (used for bulk
# writes). They must wrap the same underlying model.
EmbedOne = t.Callable[[str], list[float]]
EmbedMany = t.Callable[[t.Sequence[str]], list[list[float]]]


# -------- DATA MODEL -----------------------------------------------------------
class MemoryRecord(BaseModel):
    """The unit of storage. Richer than (key, value): the extra fields are
    the quality levers (confidence/source) and the self-cleaning lever (TTL)."""

    key: str
    category: str
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str | None = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or datetime.now(timezone.utc)) >= self.expires_at


class RecallQuery(BaseModel):
    """A relevance request. `text` is the current turn/task; backends rank
    against it. Empty text = "give me the most salient facts" (cold start)."""

    text: str | None = None
    category: str | None = None
    limit: int = 8
    min_confidence: float = 0.0


class MergePolicy(BaseModel):
    """Contract-level dedup so every backend behaves the same. Backends only
    provide the similarity primitive (the embedder); the policy lives here."""

    similarity_threshold: float = 0.85
    exact_match_wins: bool = True
    keep_higher_confidence: bool = True


class MemoryToolMode(str, Enum):
    """Which memory tools are exposed to the LLM.

    - NONE      : no tools; memory is read-only background context.
    - READ_ONLY : list + search.
    - FULL      : list + search + update + delete.
    """

    NONE = "none"
    READ_ONLY = "read_only"
    FULL = "full"


# -------- CONTRACT -----------------------------------------------------------
class CoreMemoryRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base for agent memory backends."""

    def __init__(
        self,
        user_id: str,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        merge_policy: MergePolicy | None = None,
        embed_one: EmbedOne | None = None,
        embed_many: EmbedMany | None = None,
        context_days: int | None = 30,
    ) -> None:
        super().__init__()
        self.user_id = self.require_type(user_id, str, "user_id")
        self.tool_mode = self.require_type(tool_mode, MemoryToolMode, "tool_mode")
        self.merge_policy = merge_policy or MergePolicy()
        # Optional. Local backends use these to rank / dedup in Python;
        # backends that delegate to a native index can leave them None.
        # embed_one: query + pairwise compare. embed_many: bulk writes.
        self._embed_one = embed_one
        self._embed_many = embed_many
        # Recency window for prompt injection. get_context() returns facts
        # touched within this many days. None = no limit. Lives here, not in
        # any one backend, because it's a policy every backend shares.
        self.context_days = context_days

    # ===== STORAGE SURFACE — implement per backend =====
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def _read_all(self) -> list[MemoryRecord]:
        """Every (non-expired) record for this user."""
        ...

    @abstractmethod
    async def _write_many(self, records: list[MemoryRecord]) -> None: ...

    @abstractmethod
    async def _delete_many(self, keys: list[str]) -> None: ...

    # ===== RETRIEVAL — default ranks in Python; override to use a native index =====
    async def recall(self, query: RecallQuery) -> list[MemoryRecord]:
        records = [r for r in await self._read_all() if not r.is_expired()]
        if query.category:
            records = [r for r in records if r.category == query.category]
        records = [r for r in records if r.confidence >= query.min_confidence]

        if not query.text or self._embed_one is None or self._embed_many is None:
            records.sort(key=lambda r: r.last_updated, reverse=True)
            return records[: query.limit]

        qv = self._embed_one(query.text)
        # One batch call for all records instead of N singular calls.
        vecs = self._embed_many([f"{r.category}\n{r.content}" for r in records])
        scored = [
            (self._cosine(qv, v), r) for v, r in zip(vecs, records)
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [r for _, r in scored[: query.limit]]

    # ===== PROMPT INJECTION — recency-windowed, no query =====
    async def get_context(self) -> list[MemoryRecord]:
        """Salient facts for prompt injection at agent build time.

        Called from prepare() BEFORE any user message exists, so there is
        no query to rank against — this returns recent, non-expired facts
        ordered newest-first, bounded by `context_days`. Relevance-based
        retrieval happens later, on demand, via the search tool (recall()),
        once the user has actually said something.
        """
        records = [r for r in await self._read_all() if not r.is_expired()]
        if self.context_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.context_days)
            records = [r for r in records if r.last_updated >= cutoff]
        records.sort(key=lambda r: r.last_updated, reverse=True)
        return records

    # ===== WRITES — batch-first, merge applied here =====
    async def upsert(self, record: MemoryRecord) -> None:
        await self.upsert_many([record])

    async def upsert_many(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        existing = await self._read_all()
        to_write, to_drop = self._apply_merge(existing, records)
        if to_drop:
            await self._delete_many(to_drop)
        await self._write_many(to_write)

    async def delete_many(self, keys: list[str]) -> None:
        await self._delete_many([k for k in keys if k])

    # ----- convenience kept for the maintenance pass in agent.py -----
    async def upsert_memory(self, *, key: str, category: str, content: str) -> None:
        await self.upsert(
            MemoryRecord(
                key=self._validate_non_empty("key", key),
                category=self._validate_non_empty("category", category),
                content=self._validate_non_empty("content", content),
            )
        )

    async def update_fact(self, key: str, value: str) -> None:
        await self.upsert(
            MemoryRecord(
                key=self._validate_non_empty("key", key),
                category="general",
                content=self._validate_non_empty("value", value),
            )
        )

    async def delete_fact(self, key: str) -> None:
        await self.delete_many([self._validate_non_empty("key", key)])

    async def list_facts(self) -> list[MemoryRecord]:
        return await self._read_all()

    # ===== MERGE POLICY =====
    def _apply_merge(
        self, existing: list[MemoryRecord], incoming: list[MemoryRecord]
    ) -> tuple[list[MemoryRecord], list[str]]:
        to_write: list[MemoryRecord] = []
        to_drop: list[str] = []
        by_key = {r.key: r for r in existing}

        for rec in incoming:
            if rec.key in by_key:               # exact key reuse → overwrite
                to_write.append(rec)
                continue
            twin = self._find_twin(rec, by_key.values())
            if twin is None:                    # genuinely new
                to_write.append(rec)
                continue
            if self.merge_policy.keep_higher_confidence and twin.confidence > rec.confidence:
                continue                        # existing wins, drop incoming
            rec = rec.model_copy(update={"key": twin.key})  # fold into existing key
            to_write.append(rec)

        return to_write, to_drop

    def _find_twin(
        self, rec: MemoryRecord, pool: t.Iterable[MemoryRecord]
    ) -> MemoryRecord | None:
        norm = self._normalize(rec.content)
        same_cat = [o for o in pool if o.category == rec.category]
        if not same_cat:
            return None

        # Exact-content match short-circuits without any embedding.
        if self.merge_policy.exact_match_wins:
            for other in same_cat:
                if self._normalize(other.content) == norm:
                    return other

        if self._embed_many is None:
            return None

        # One batch call: [incoming, *pool] embedded together.
        texts = [f"{rec.category}\n{rec.content}"] + [
            f"{o.category}\n{o.content}" for o in same_cat
        ]
        vecs = self._embed_many(texts)
        rec_vec, pool_vecs = vecs[0], vecs[1:]

        best: MemoryRecord | None = None
        best_score = 0.0
        for other, ov in zip(same_cat, pool_vecs):
            s = self._cosine(rec_vec, ov)
            if s > best_score:
                best, best_score = other, s
        if best is not None and best_score >= self.merge_policy.similarity_threshold:
            return best
        return None

    # ===== TOOLS EXPOSED TO THE LLM =====
    def as_tools(self) -> list[CoreTool]:
        if self.tool_mode == MemoryToolMode.NONE:
            return []
        if self.tool_mode == MemoryToolMode.READ_ONLY:
            return [self._build_list_tool(), self._build_search_tool()]
        return [
            self._build_list_tool(),
            self._build_search_tool(),
            self._build_update_tool(),
            self._build_delete_tool(),
        ]

    @property
    def tools(self) -> list[CoreTool]:
        return self.as_tools()

    def _build_list_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool

        async def list_memories() -> list[dict[str, t.Any]]:
            """List every fact currently stored about the user."""
            await self._ensure_connected()
            return [r.model_dump(exclude_none=True) for r in await self.list_facts()]

        return FunctionAsTool(
            list_memories,
            name="list_memories",
            description="List every fact currently stored about the user across past sessions.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_search_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool

        async def search_memories(
            query: str, limit: int = 5, category: str | None = None
        ) -> list[dict[str, t.Any]]:
            """Search durable user memories by semantic meaning."""
            await self._ensure_connected()
            results = await self.recall(RecallQuery(text=query, limit=limit, category=category))
            return [r.model_dump(exclude_none=True) for r in results]

        return FunctionAsTool(
            search_memories,
            name="search_memories",
            description=(
                "Search durable memories about this user by meaning. Use before "
                "creating a memory to avoid duplicating or contradicting existing facts."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_update_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool

        async def update_memory(
            key: str, category: str, content: str, confidence: float = 1.0
        ) -> str:
            """Create or update a durable memory about the user.

            Args:
                key: Stable short identifier for this fact.
                category: preference, profile, project, goal, constraint, relationship.
                content: The durable user fact to remember.
                confidence: 0..1 — lower it for inferred rather than stated facts.
            """
            await self._ensure_connected()
            await self.upsert(
                MemoryRecord(
                    key=key, category=category, content=content,
                    confidence=confidence, source="agent",
                )
            )
            return f"Memory saved: {key}"

        return FunctionAsTool(
            update_memory,
            name="update_memory",
            description=(
                "Store or overwrite a durable fact about the user. Provide a stable "
                "key, a category, concise content, and a confidence. Use for long-lived "
                "preferences, profile, goals, projects, constraints, relationships. Not "
                "for transient conversation details."
            ),
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )

    def _build_delete_tool(self) -> CoreTool:
        from ..tools import FunctionAsTool

        async def delete_memory(key: str) -> str:
            """Delete a previously stored fact about the user."""
            await self._ensure_connected()
            await self.delete_fact(key)
            return f"Memory deleted: {key}"

        return FunctionAsTool(
            delete_memory,
            name="delete_memory",
            description=(
                "Delete a stored fact about the user. Use when the user asks to forget "
                "something, or when newer information makes a stored fact obsolete."
            ),
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )

    # ===== HELPERS =====
    def _validate_non_empty(self, field: str, value: str) -> str:
        from ..errors.memory import MemoryError

        if not isinstance(value, str):
            raise MemoryError.invalid_type(field, "str", type(value).__name__)
        clean = value.strip()
        if not clean:
            raise MemoryError.missing(field)
        return clean

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0