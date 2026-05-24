"""
Filesystem-backed ``RunContextStore``.

Stores each run as a pretty-printed JSON file under ``base_path``.
Suitable for single-process deployments (CLI tools, dev servers,
single worker) and for debugging — the files are human-readable and
can be inspected with any editor.

Atomicity: writes go to ``<run_id>.json.tmp`` first and are renamed
into place. On POSIX, ``os.rename`` is atomic — a process crash mid-
write leaves the prior version intact.

Concurrency: last-write-wins. Two processes calling ``save`` for the
same ``run_id`` simultaneously can produce a final file from either
write — the framework makes no guarantee about which. Production
deployments that span multiple processes should use a backend with
explicit concurrency semantics (Postgres row locks, Redis SETNX, etc.).

I/O: synchronous file ops dispatched via ``asyncio.to_thread`` to keep
the public surface async without pulling in ``aiofiles``. JSON files
are small, so the threadpool overhead is negligible.
"""

from __future__ import annotations

import os
import logging
import asyncio
from pathlib import Path

from ..loggers import ScopedLogger
from ..types.run_context import RunContext

from .core import RunContextStore, validate_run_id


logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["FileSystemRunContextStore"])


class FileSystemRunContextStore(RunContextStore):
    """``RunContextStore`` backed by a directory of JSON files.

    Each ``RunContext`` lives at ``<base_path>/<run_id>.json``. The
    directory is created automatically on construction.
    """

    _SUFFIX = ".json"
    _TMP_SUFFIX = ".json.tmp"
    _ENCODING = "utf-8"

    def __init__(self, base_path: str | Path) -> None:
        """Initialize the store.

        Args:
            base_path: Directory where run files live. Created if it
                does not exist. Must be writable.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # -------- PATH HELPERS -----------------------------------------------------------
    def _path_for(self, run_id: str) -> Path:
        return self.base_path / f"{run_id}{self._SUFFIX}"

    def _tmp_path_for(self, run_id: str) -> Path:
        return self.base_path / f"{run_id}{self._TMP_SUFFIX}"

    # -------- CONTRACT -----------------------------------------------------------
    async def save(self, run_id: str, ctx: RunContext) -> None:
        """Atomically write the context to ``<run_id>.json``.

        Implementation: serialize to JSON, write to a sibling ``.tmp``
        file, then ``os.rename`` over the destination. The rename is
        atomic on POSIX so concurrent readers never see a half-written
        file.
        """
        validate_run_id(run_id)
        payload = ctx.model_dump_json(indent=2)
        await asyncio.to_thread(self._write_atomic, run_id, payload)
        log.child(run_id=run_id).info("RunContext saved")

    async def load(self, run_id: str) -> RunContext | None:
        """Read and deserialize the context, or return ``None`` if missing."""
        validate_run_id(run_id)
        payload = await asyncio.to_thread(self._read, run_id)
        if payload is None:
            return None
        try:
            return RunContext.model_validate_json(payload)
        except Exception as e:
            # A malformed file is more dangerous than a missing one —
            # the caller probably wants to know rather than silently
            # starting a fresh context.
            log.child(run_id=run_id).error(
                "Failed to deserialize RunContext", error=str(e)
            )
            raise

    async def delete(self, run_id: str) -> None:
        """Remove the stored context. Idempotent."""
        validate_run_id(run_id)
        await asyncio.to_thread(self._unlink, run_id)
        log.child(run_id=run_id).info("RunContext deleted (or absent)")

    async def list(self) -> list[str]:
        """Enumerate run ids by listing the directory."""
        return await asyncio.to_thread(self._list_sync)

    # -------- SYNC I/O (run via to_thread) -----------------------------------------------------------
    def _write_atomic(self, run_id: str, payload: str) -> None:
        tmp = self._tmp_path_for(run_id)
        dest = self._path_for(run_id)
        # Best-effort cleanup of a stale tmp from a prior crash.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        tmp.write_text(payload, encoding=self._ENCODING)
        os.replace(tmp, dest)  # atomic on POSIX, best-effort on Windows

    def _read(self, run_id: str) -> str | None:
        path = self._path_for(run_id)
        try:
            return path.read_text(encoding=self._ENCODING)
        except FileNotFoundError:
            return None

    def _unlink(self, run_id: str) -> None:
        self._path_for(run_id).unlink(missing_ok=True)

    def _list_sync(self) -> list[str]:
        if not self.base_path.exists():
            return []
        suffix_len = len(self._SUFFIX)
        return sorted(
            p.name[:-suffix_len]
            for p in self.base_path.iterdir()
            if p.is_file() and p.name.endswith(self._SUFFIX)
        )
