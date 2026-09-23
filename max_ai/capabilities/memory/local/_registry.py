"""Filesystem-backed memory registry, one JSON file per (user, session)."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    ) -> None:
        if base_path is None:
            raise ValueError("LocalMemoryRegistry needs a base_path")
        self.base_path: Path = Path(base_path).expanduser().resolve()
        super().__init__(
            user_id=user_id, session_id=session_id,
            tool_mode=tool_mode, context_days=context_days,
        )

    def _validate_scope(self) -> None:
        if not _SAFE_ID_RE.match(self.user_id) or not _SAFE_ID_RE.match(self.session_id):
            raise ValueError(
                "user_id/session_id must match "
                f"{_SAFE_ID_RE.pattern!r} to be safe as filesystem paths"
            )

    def _to_config(self) -> LocalMemoryRegistryConfig:
        return LocalMemoryRegistryConfig(
            user_id=self.user_id,
            session_id=self.session_id,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
            context_days=self.context_days,
        )

    @classmethod
    def _from_config(cls, config: LocalMemoryRegistryConfig) -> "LocalMemoryRegistry":
        return cls(
            user_id=config.user_id,
            session_id=config.session_id,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
            context_days=config.context_days,
        )

    @property
    def _memory_dir(self) -> Path:
        return self.base_path / "memory"

    @property
    def _user_dir(self) -> Path:
        return self._memory_dir / self.user_id

    def _session_file(self, session_id: str) -> Path:
        return self._user_dir / f"{session_id}.json"

    async def connect(self) -> None:
        # Per-user folders are created on first write (see _write_store).
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        return None

    # -------- STORE I/O -----------------------------------------------------------
    @staticmethod
    def _load_store(path: Path) -> dict[str, MemoryRecord]:
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
        await self._ensure_connected()
        return list(self._load_store(self._session_file(self.session_id)).values())

    async def _write_memory(self, record: MemoryRecord) -> bool:
        await self._ensure_connected()
        path = self._session_file(self.session_id)
        store = self._load_store(path)
        created = record.category not in store
        store[record.category] = record
        self._write_store(path, store)
        return created

    async def _delete_memory(self, category: str) -> bool:
        await self._ensure_connected()
        path = self._session_file(self.session_id)
        store = self._load_store(path)
        if category not in store:
            return False
        del store[category]
        self._write_store(path, store)
        return True

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        await self._ensure_connected()
        needle = text.strip().lower()
        results: list[MemorySearchResult] = []
        for path in sorted(self._user_dir.glob("*.json")):
            session_id = path.stem
            for record in self._load_store(path).values():
                if needle in record.category.lower() or needle in record.memory.lower():
                    results.append(
                        MemorySearchResult(
                            category=record.category,
                            memory=record.memory,
                            updated=record.updated,
                            session_id=session_id,
                        )
                    )
        return results
