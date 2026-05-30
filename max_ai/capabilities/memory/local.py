"""Filesystem-backed memory registry."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ...base.memory import CoreMemoryRegistry, MemoryRecord, MemoryToolMode


class LocalMemoryRegistryConfig(BaseModel):
    user_id: str
    base_path: str
    tool_mode: MemoryToolMode = MemoryToolMode.FULL


class LocalMemoryRegistry(CoreMemoryRegistry):
    component_schema = LocalMemoryRegistryConfig
    component_type = "memory"

    """Filesystem-backed implementation of ``CoreMemoryRegistry``.

    Layout::

        {base_path}/
            memory/
                {user_id}.json          # {key: MemoryRecord, ...}

    The file is read on every operation. This keeps the on-disk state
    authoritative and makes it simple to inspect or edit during local dev.
    """

    def __init__(
        self,
        user_id: str,
        base_path: str | Path,
        tool_mode: MemoryToolMode = MemoryToolMode.FULL,
    ) -> None:
        super().__init__(user_id=user_id, tool_mode=tool_mode)
        self.base_path: Path = Path(base_path).expanduser().resolve()

    def _to_config(self) -> LocalMemoryRegistryConfig:
        return LocalMemoryRegistryConfig(
            user_id=self.user_id,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
        )

    @classmethod
    def _from_config(cls, config: LocalMemoryRegistryConfig) -> "LocalMemoryRegistry":
        return cls(
            user_id=config.user_id,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
        )

    @property
    def _memory_dir(self) -> Path:
        return self.base_path / "memory"

    @property
    def _user_file(self) -> Path:
        return self._memory_dir / f"{self.user_id}.json"

    async def connect(self) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        return None

    async def _read_all(self) -> list[MemoryRecord]:
        await self._ensure_connected()
        return list(self._load_store().values())

    async def _write_many(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        await self._ensure_connected()
        store = self._load_store()
        for record in records:
            store[record.key] = record
        self._write_all(store)

    async def _delete_many(self, keys: list[str]) -> None:
        if not keys:
            return
        await self._ensure_connected()
        store = self._load_store()
        changed = False
        for key in keys:
            if key in store:
                del store[key]
                changed = True
        if changed:
            self._write_all(store)

    def _load_store(self) -> dict[str, MemoryRecord]:
        path = self._user_file
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
        for key, payload in raw.items():
            try:
                if isinstance(payload, dict) and "key" not in payload:
                    payload = {**payload, "key": key}
                record = MemoryRecord.model_validate(payload)
                result[record.key] = record
            except Exception as e:
                raise ValueError(
                    f"Memory file {path}: entry {key!r} is not a valid "
                    f"MemoryRecord: {e}"
                ) from e
        return result

    def _write_all(self, store: dict[str, MemoryRecord]) -> None:
        path = self._user_file
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            key: json.loads(record.model_dump_json(exclude_none=True))
            for key, record in store.items()
        }
        text = json.dumps(serializable, indent=2, ensure_ascii=False)

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
