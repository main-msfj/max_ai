"""Usage as JSON files: ``<base>/<user_id>.json`` → ``{period_key: usage}``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from ....base.quota_store import CoreQuotaStore
from ....core.model.quota import QuotaUsage
from ._model import LocalQuotaStoreConfig


class LocalQuotaStore(CoreQuotaStore):
    """One small file per user. ``add`` is atomic within one process (a lock
    per user, atomic file writes); several processes sharing the folder
    need a database-backed store instead."""

    component_schema = LocalQuotaStoreConfig
    component_provider_override = "max_ai.capabilities.quota_store.local.LocalQuotaStore"

    def __init__(self, base_path: str | Path, keep_periods: int = 60) -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.keep_periods = keep_periods
        self._locks: dict[str, asyncio.Lock] = {}

    def _to_config(self) -> LocalQuotaStoreConfig:
        return LocalQuotaStoreConfig(base_path=str(self.base_path), keep_periods=self.keep_periods)

    @classmethod
    def _from_config(cls, config: LocalQuotaStoreConfig) -> LocalQuotaStore:
        return cls(base_path=config.base_path, keep_periods=config.keep_periods)

    def _path(self, user_id: str) -> Path:
        return self.base_path / f"{user_id}.json"

    # -------- BACKEND HOOKS -----------------------------------------------------------
    async def _usage(self, user_id: str, period_key: str) -> QuotaUsage:
        periods = await asyncio.to_thread(_read, self._path(user_id))
        return QuotaUsage.model_validate(periods.get(period_key, {}))

    async def _add(self, user_id: str, period_key: str, delta: QuotaUsage) -> QuotaUsage:
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            path = self._path(user_id)
            periods = await asyncio.to_thread(_read, path)
            total = QuotaUsage.model_validate(periods.get(period_key, {})) + delta
            periods[period_key] = total.model_dump()
            # Keys sort by date within a kind (day:/month:), newest last.
            kept = dict(sorted(periods.items())[-self.keep_periods:])
            await asyncio.to_thread(_write_atomic, path, json.dumps(kept))
            return total


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
