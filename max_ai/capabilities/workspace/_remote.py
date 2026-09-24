"""Shared sync logic for workspaces whose files live in object storage.

Tools always work on a local copy (``<cache>/<user>/workspace``). The Agent
calls ``download`` when a run starts and ``upload`` when it ends, so a new
process (a serverless invocation, another server) sees the user's files.

Only differences move: a manifest remembers each file's SHA-256 as of the
last sync, and objects carry that hash as metadata. Scratchpads and skills
stay local (throwaway / materialized per run). Two processes writing the
same file at once: the last upload wins.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import typing as t
from abc import abstractmethod
from pathlib import Path

from ...base.workspace import WorkspaceBase

_MANIFEST = ".maxai-sync.json"


class RemoteObject(t.NamedTuple):
    """RemoteObject represents structured data used by the capability system."""
    key: str  # relative to the user's workspace prefix
    sha256: str | None


class RemoteWorkspace(WorkspaceBase):
    """A workspace mirrored to a remote store; subclasses move bytes."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize ``RemoteWorkspace``.

Parameters
----------
cache_dir : str | Path | None
    Value supplied for ``cache_dir``."""
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        root = cache_dir or Path(tempfile.gettempdir()) / "maxai-workspaces" / self._cache_name()
        super().__init__(root)
        self._locks: dict[str, asyncio.Lock] = {}

    def _cache_name(self) -> str:
        """Stable local folder per remote location."""
        return hashlib.sha256(self._location().encode()).hexdigest()[:16]

    # -------- BACKEND HOOKS -----------------------------------------------------------
    @abstractmethod
    def _location(self) -> str:
        """Where the files live (URL + container/bucket); names the cache."""

    @abstractmethod
    async def _list(self, prefix: str) -> list[RemoteObject]:
        """Perform the internal ``list`` operation for ``RemoteWorkspace``.

Parameters
----------
prefix : str
        Value supplied for ``prefix``."""
        ...

    @abstractmethod
    async def _get(self, key: str) -> bytes:
        """Perform the internal ``get`` operation for ``RemoteWorkspace``.

Parameters
----------
key : str
        Value supplied for ``key``."""
        ...

    @abstractmethod
    async def _put(self, key: str, data: bytes, sha256: str) -> None:
        """Perform the internal ``put`` operation for ``RemoteWorkspace``.

Parameters
----------
key : str
    Value supplied for ``key``.
data : bytes
    Value supplied for ``data``.
sha256 : str
        Value supplied for ``sha256``."""
        ...

    @abstractmethod
    async def _delete(self, key: str) -> None:
        """Perform the internal ``delete`` operation for ``RemoteWorkspace``.

Parameters
----------
key : str
        Value supplied for ``key``."""
        ...

    # -------- SYNC -----------------------------------------------------------
    @staticmethod
    def _prefix(user_id: str) -> str:
        """Perform the internal ``prefix`` operation for ``RemoteWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        return f"{user_id}/workspace/"

    def _local(self, user_id: str) -> Path:
        """Perform the internal ``local`` operation for ``RemoteWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        return self.materialize(user_id).workspace_dir

    def _manifest(self, user_id: str) -> Path:
        """Perform the internal ``manifest`` operation for ``RemoteWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        return Path(self.base_root) / user_id / _MANIFEST

    def _read_manifest(self, user_id: str) -> dict[str, str]:
        """Perform the internal ``read manifest`` operation for ``RemoteWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``."""
        try:
            return json.loads(self._manifest(user_id).read_text())
        except FileNotFoundError:
            return {}

    def _write_manifest(self, user_id: str, manifest: dict[str, str]) -> None:
        """Perform the internal ``write manifest`` operation for ``RemoteWorkspace``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
manifest : dict[str, str]
    Value supplied for ``manifest``."""
        self._manifest(user_id).write_text(json.dumps(manifest, sort_keys=True))

    @staticmethod
    def _scan(root: Path) -> dict[str, str]:
        """Perform the internal ``scan`` operation for ``RemoteWorkspace``.

Parameters
----------
root : Path
    Value supplied for ``root``."""
        files = {}
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and not path.name.startswith(".maxai-"):
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return files

    async def download(self, user_id: str, conversation_id: str | None = None) -> None:
        """Bring the local copy up to date with the remote files."""
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            root = self._local(user_id)
            prefix = self._prefix(user_id)
            remote = {obj.key: obj.sha256 for obj in await self._list(prefix)}
            manifest = self._read_manifest(user_id)
            local = await asyncio.to_thread(self._scan, root)
            for key, sha in remote.items():
                if sha is None or local.get(key) != sha:
                    data = await self._get(prefix + key)
                    target = root / key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    manifest[key] = hashlib.sha256(data).hexdigest()
            for key in [k for k in manifest if k not in remote]:
                # Deleted remotely since our last sync (never a file we made locally).
                if local.get(key) == manifest[key]:
                    (root / key).unlink(missing_ok=True)
                manifest.pop(key)
            self._write_manifest(user_id, manifest)

    async def upload(self, user_id: str, conversation_id: str | None = None) -> None:
        """Send the files that changed locally; delete the ones removed."""
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            root = self._local(user_id)
            prefix = self._prefix(user_id)
            manifest = self._read_manifest(user_id)
            local = await asyncio.to_thread(self._scan, root)
            for key, sha in local.items():
                if manifest.get(key) != sha:
                    await self._put(prefix + key, (root / key).read_bytes(), sha)
                    manifest[key] = sha
            for key in [k for k in manifest if k not in local]:
                await self._delete(prefix + key)
                manifest.pop(key)
            self._write_manifest(user_id, manifest)


__all__ = ["RemoteObject", "RemoteWorkspace"]
