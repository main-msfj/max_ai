"""Filesystem-backed memory registry, one JSON file per (user, session)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ....base.embedding import DEFAULT_EMBEDDING, CoreEmbedding
from ....base.memory import (
    CoreMemoryRegistry,
    MemoryRecord,
    MemorySearchResult,
    MemoryToolMode,
)
from ._model import LocalMemoryRegistryConfig

# user_id/session_id become path segments (a directory and a filename) —
# restricted so neither can escape base_path or collide with another user.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class LocalMemoryRegistry(CoreMemoryRegistry):
    """Filesystem-backed implementation of ``CoreMemoryRegistry``.

    Layout::

        {base_path}/memory/
            {user_id}/
                {session_id}.json      # {category: MemoryRecord, ...}

    One file per session (not per user) because the contract is
    session-scoped: ``_read_session`` only ever needs this session's
    file, and ``_search_memory`` scans every other session file under
    the same user directory. Each file is read/written whole — simple
    and correct at the scale a single user's session memory reaches;
    no separate index is maintained.
    """

    component_provider_override = "max_ai.capabilities.memory.local.LocalMemoryRegistry"
    component_schema = LocalMemoryRegistryConfig
    component_type = "memory"

    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        base_path: str | Path | None = None,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
        *,
        context_days: int | None = 30,
        embedding: CoreEmbedding | None = DEFAULT_EMBEDDING,
    ) -> None:
        """Initialize ``LocalMemoryRegistry``.

Parameters
----------
user_id : str | None
    Value supplied for ``user_id``.
session_id : str | None
    Value supplied for ``session_id``.
base_path : str | Path | None
    Value supplied for ``base_path``.
tool_mode : MemoryToolMode
    Value supplied for ``tool_mode``.
context_days : int | None
    Value supplied for ``context_days``.
embedding : CoreEmbedding | None
    Value supplied for ``embedding``."""
        if base_path is None:
            raise ValueError("LocalMemoryRegistry needs a base_path")
        self.base_path: Path = Path(base_path).expanduser().resolve()
        super().__init__(
            user_id=user_id, session_id=session_id,
            tool_mode=tool_mode, context_days=context_days, embedding=embedding,
        )

    def _validate_scope(self) -> None:
        """Perform the internal ``validate scope`` operation for ``LocalMemoryRegistry``."""
        if not _SAFE_ID_RE.match(self.user_id) or not _SAFE_ID_RE.match(self.session_id):
            raise ValueError(
                "user_id/session_id must match "
                f"{_SAFE_ID_RE.pattern!r} to be safe as filesystem paths"
            )

    def _to_config(self) -> LocalMemoryRegistryConfig:
        """Build the serializable configuration for ``LocalMemoryRegistry``."""
        return LocalMemoryRegistryConfig(
            user_id=self.user_id,
            session_id=self.session_id,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
            context_days=self.context_days,
            embedding=self.embedding.serialize().model_dump(exclude_none=True) if self.embedding else None,
        )

    @classmethod
    def _from_config(cls, config: LocalMemoryRegistryConfig) -> "LocalMemoryRegistry":
        """Create an instance from its configuration for ``LocalMemoryRegistry``.

Parameters
----------
config : LocalMemoryRegistryConfig
    Value supplied for ``config``."""
        return cls(
            user_id=config.user_id,
            session_id=config.session_id,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
            context_days=config.context_days,
            embedding=CoreEmbedding.deserialize(config.embedding) if config.embedding else None,
        )

    @property
    def _memory_dir(self) -> Path:
        """Perform the internal ``memory dir`` operation for ``LocalMemoryRegistry``."""
        return self.base_path / "memory"

    @property
    def _user_dir(self) -> Path:
        """Perform the internal ``user dir`` operation for ``LocalMemoryRegistry``."""
        return self._memory_dir / self.user_id

    def _session_file(self, session_id: str) -> Path:
        """Perform the internal ``session file`` operation for ``LocalMemoryRegistry``.

Parameters
----------
session_id : str
    Value supplied for ``session_id``."""
        return self._user_dir / f"{session_id}.json"

    async def connect(self) -> None:
        # Per-user folders are created on first write (see _write_store).
        """Open required resources for ``LocalMemoryRegistry``."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        """Release resources held for ``LocalMemoryRegistry``."""
        return None

    # -------- STORE I/O -----------------------------------------------------------
    @staticmethod
    def _load_store(path: Path) -> dict[str, MemoryRecord]:
        """Perform the internal ``load store`` operation for ``LocalMemoryRegistry``.

Parameters
----------
path : Path
    Value supplied for ``path``."""
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Memory file {path} is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(
                f"Memory file {path} must contain a JSON object at the "
                f"top level, got {type(raw).__name__}"
            )
        result: dict[str, MemoryRecord] = {}
        for category, payload in raw.items():
            try:
                record = MemoryRecord.model_validate(payload)
            except Exception as e:
                raise ValueError(
                    f"Memory file {path}: entry {category!r} is not a valid "
                    f"MemoryRecord: {e}"
                ) from e
            result[record.category] = record
        return result

    @staticmethod
    def _write_store(path: Path, store: dict[str, MemoryRecord]) -> None:
        """Perform the internal ``write store`` operation for ``LocalMemoryRegistry``.

Parameters
----------
path : Path
    Value supplied for ``path``.
store : dict[str, MemoryRecord]
    Value supplied for ``store``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            category: json.loads(record.model_dump_json())
            for category, record in store.items()
        }
        text = json.dumps(serializable, indent=2, ensure_ascii=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    # -------- CONTRACT -----------------------------------------------------------
    async def _read_session(self) -> list[MemoryRecord]:
        """Perform the internal ``read session`` operation for ``LocalMemoryRegistry``."""
        await self._ensure_connected()
        return list(self._load_store(self._session_file(self.session_id)).values())

    async def _read_user(self) -> list[MemorySearchResult]:
        """Every session file of this user."""
        await self._ensure_connected()
        return [
            MemorySearchResult(**record.model_dump(), session_id=path.stem)
            for path in sorted(self._user_dir.glob("*.json"))
            for record in self._load_store(path).values()
        ]

    async def _write_memory(self, record: MemoryRecord) -> bool:
        """Perform the internal ``write memory`` operation for ``LocalMemoryRegistry``.

Parameters
----------
record : MemoryRecord
    Value supplied for ``record``."""
        await self._ensure_connected()
        path = self._session_file(self.session_id)
        store = self._load_store(path)
        created = record.category not in store
        store[record.category] = record
        self._write_store(path, store)
        return created

    async def _delete_memory(self, category: str) -> bool:
        """Perform the internal ``delete memory`` operation for ``LocalMemoryRegistry``.

Parameters
----------
category : str
    Value supplied for ``category``."""
        await self._ensure_connected()
        path = self._session_file(self.session_id)
        store = self._load_store(path)
        if category not in store:
            return False
        del store[category]
        self._write_store(path, store)
        return True

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        """Other sessions' memories: by meaning with an embedding, else by words."""
        await self._ensure_connected()
        needle = text.strip().lower()
        candidates = [
            MemorySearchResult(category=record.category, memory=record.memory,
                               updated=record.updated, session_id=path.stem)
            for path in sorted(self._user_dir.glob("*.json")) if path.stem != self.session_id
            for record in self._load_store(path).values()
        ]
        if self.embedding is not None:
            return await self._rank_by_meaning(text, candidates)
        return [c for c in candidates if needle in c.category.lower() or needle in c.memory.lower()]
