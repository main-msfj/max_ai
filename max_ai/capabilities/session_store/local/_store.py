"""Sessions as JSON files: ``<base>/<user_id>/<session_id>.json``."""

from __future__ import annotations

import asyncio
import os
import typing as t
from pathlib import Path

from ....base.session_store import CoreSessionStore
from ....core.model.session import SessionInfo
from ....types.run_context import RunContext
from ._model import LocalSessionStoreConfig


class LocalSessionStore(CoreSessionStore):
    """One folder per user, two files per session: the full ``RunContext``
    and a small ``.meta.json`` so listing never reads whole conversations.
    Writes are atomic (temp file + rename): readers never see half a file.
    """

    component_schema = LocalSessionStoreConfig
    component_provider_override = "maxai.session_store.LocalSessionStore"

    def __init__(self, base_path: str | Path) -> None:
        """Initialize ``LocalSessionStore``.

Parameters
----------
base_path : str | Path
    Value supplied for ``base_path``."""
        super().__init__()
        self.base_path = Path(base_path)

    def _to_config(self) -> LocalSessionStoreConfig:
        """Build the serializable configuration for ``LocalSessionStore``."""
        return LocalSessionStoreConfig(base_path=str(self.base_path))

    @classmethod
    def _from_config(cls, config: LocalSessionStoreConfig) -> "LocalSessionStore":
        """Create an instance from its configuration for ``LocalSessionStore``.

Parameters
----------
config : LocalSessionStoreConfig
    Value supplied for ``config``."""
        return cls(base_path=config.base_path)

    def _paths(self, user_id: str, session_id: str) -> tuple[Path, Path]:
        """Perform the internal ``paths`` operation for ``LocalSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
session_id : str
    Value supplied for ``session_id``."""
        folder = self.base_path / user_id
        return folder / f"{session_id}.json", folder / f"{session_id}.meta.json"

    # -------- BACKEND HOOKS -----------------------------------------------------------
    async def _load(self, user_id: str, session_id: str) -> RunContext | None:
        """Perform the internal ``load`` operation for ``LocalSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
session_id : str
    Value supplied for ``session_id``."""
        path, _ = self._paths(user_id, session_id)
        payload = await asyncio.to_thread(_read, path)
        return None if payload is None else RunContext.model_validate_json(payload)

    async def _save(self, ctx: RunContext, info: SessionInfo) -> None:
        """Perform the internal ``save`` operation for ``LocalSessionStore``.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``.
info : SessionInfo
    Value supplied for ``info``."""
        path, meta = self._paths(ctx.user_id, ctx.session_id or "")
        await asyncio.to_thread(_write_atomic, path, ctx.model_dump_json())
        await asyncio.to_thread(_write_atomic, meta, info.model_dump_json())

    async def _list(self, user_id: str) -> t.Iterable[SessionInfo]:
        """Perform the internal ``list`` operation for ``LocalSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        folder = self.base_path / user_id
        payloads = await asyncio.to_thread(_read_all, folder, "*.meta.json")
        return [SessionInfo.model_validate_json(p) for p in payloads]

    async def _delete(self, user_id: str, session_id: str) -> bool:
        """Perform the internal ``delete`` operation for ``LocalSessionStore``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
session_id : str
    Value supplied for ``session_id``."""
        path, meta = self._paths(user_id, session_id)
        existed = path.exists()
        await asyncio.to_thread(meta.unlink, missing_ok=True)
        await asyncio.to_thread(path.unlink, missing_ok=True)
        return existed


def _write_atomic(path: Path, payload: str) -> None:
    """Perform the internal ``write atomic`` operation.

Parameters
----------
path : Path
    Value supplied for ``path``.
payload : str
    Value supplied for ``payload``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> str | None:
    """Perform the internal ``read`` operation.

Parameters
----------
path : Path
    Value supplied for ``path``."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _read_all(folder: Path, pattern: str) -> list[str]:
    """Perform the internal ``read all`` operation.

Parameters
----------
folder : Path
    Value supplied for ``folder``.
pattern : str
    Value supplied for ``pattern``."""
    if not folder.is_dir():
        return []
    return [p.read_text(encoding="utf-8") for p in folder.glob(pattern) if p.is_file()]
