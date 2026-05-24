"""
Filesystem-backed memory registry.

Stores user facts as a single JSON file per user under
``{base_path}/memory/{user_id}.json``. The file holds a flat mapping
from key → MemoryBlock — keys are stable identifiers (e.g.
``"user_identity"``, ``"language"``, ``"project"``) and double as the
block's category.

Not concurrency-safe: simultaneous writers from different processes can
clobber each other's changes. Intended for local development and tests,
not multi-process production workloads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ...base.memory import CoreMemoryRegistry, MemoryToolMode
from ...core import MemoryBlock


class LocalMemoryRegistry(CoreMemoryRegistry):
    """Filesystem-backed implementation of ``CoreMemoryRegistry``.

    Layout::

        {base_path}/
            memory/
                {user_id}.json          # {key: MemoryBlock, ...}

    The file is read on every operation — no caching. This keeps the
    on-disk state authoritative and makes it trivial to inspect or
    edit by hand for debugging.
    """

    def __init__(
        self,
        user_id: str,
        base_path: str | Path,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
    ) -> None:
        super().__init__(user_id=user_id, tool_mode=tool_mode)
        self.base_path: Path = Path(base_path).expanduser().resolve()

    # -------- PATH HELPERS -----------------------------------------------------------
    @property
    def _memory_dir(self) -> Path:
        return self.base_path / "memory"

    @property
    def _user_file(self) -> Path:
        return self._memory_dir / f"{self.user_id}.json"

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Ensure the memory directory exists. The user file itself is
        only created on first write — readers tolerate its absence."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        return None

    # -------- READ OPERATIONS -----------------------------------------------------------
    async def get_context(self) -> list[MemoryBlock]:
        """Return every stored fact for this user.

        Empty list when the user has no memory file yet — first call
        for a new user is not an error.
        """
        await self._ensure_connected()
        return list(self._read_all().values())

    async def list_facts(self) -> list[MemoryBlock]:
        """Same as ``get_context`` for the local backend.

        Both surfaces exist on the base contract because remote
        backends may want to expose a richer or filtered view via
        ``get_context`` (e.g. relevance-ranked) while keeping
        ``list_facts`` as the raw dump.
        """
        return await self.get_context()

    # -------- WRITE OPERATIONS -----------------------------------------------------------
    async def update_fact(self, key: str, value: str) -> None:
        """Create or overwrite the memory entry under ``key``.

        ``key`` doubles as the ``MemoryBlock.category`` and as the
        dict key on disk. Empty or whitespace-only keys are rejected.
        """
        await self._ensure_connected()
        clean_key = self._validate_key(key)
        self.require_type(value, str, "value")

        store = self._read_all()
        store[clean_key] = MemoryBlock(
            category=clean_key,
            content=value,
            last_updated=datetime.now(timezone.utc),
        )
        self._write_all(store)

    async def delete_fact(self, key: str) -> None:
        """Remove ``key`` from the store. Silent no-op if absent."""
        await self._ensure_connected()
        clean_key = self._validate_key(key)

        store = self._read_all()
        if clean_key in store:
            del store[clean_key]
            self._write_all(store)

    # -------- INTERNALS -----------------------------------------------------------
    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        clean = key.strip()
        if not clean:
            raise ValueError("key must be a non-empty, non-whitespace string")
        return clean

    def _read_all(self) -> dict[str, MemoryBlock]:
        """Load the user's full memory dict from disk.

        Returns an empty dict if the file doesn't exist yet. A corrupt
        or non-conforming file fails loud — the user finds out
        immediately instead of silently losing data.
        """
        path = self._user_file
        if not path.is_file():
            return {}

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Memory file {path} is not valid JSON: {e}"
            ) from e

        if not isinstance(raw, dict):
            raise ValueError(
                f"Memory file {path} must contain a JSON object at the "
                f"top level, got {type(raw).__name__}"
            )

        result: dict[str, MemoryBlock] = {}
        for key, payload in raw.items():
            try:
                result[key] = MemoryBlock.model_validate(payload)
            except Exception as e:
                raise ValueError(
                    f"Memory file {path}: entry {key!r} is not a valid "
                    f"MemoryBlock: {e}"
                ) from e
        return result

    def _write_all(self, store: dict[str, MemoryBlock]) -> None:
        """Persist the entire store back to disk.

        Writes to a temp file and renames atomically so a crash
        mid-write doesn't leave the user's memory file truncated.
        """
        path = self._user_file
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            key: json.loads(block.model_dump_json())
            for key, block in store.items()
        }
        text = json.dumps(serializable, indent=2, ensure_ascii=False)

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
