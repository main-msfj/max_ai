"""Workspace facade that synchronizes committed local file changes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .filesystem import UserFileSystem
from .sync import MAX_BYTES, WorkspaceSync, _error_message


class ManagedFileSystem(UserFileSystem):
    def __init__(self, root: str | os.PathLike[str], artifact_store: Any = None):
        super().__init__(root)
        try:
            self.sync: WorkspaceSync | None = WorkspaceSync(self.root, store=artifact_store)
            self._sync_init_error: str | None = None
        except Exception:
            self.sync = None
            self._sync_init_error = "workspace synchronization state unavailable"

    def refresh(self, user: str) -> dict[str, Any]:
        if self.sync is None:
            return {"items": [], "truncated": False, "state": "error", "error": self._sync_init_error}
        return self.sync.reconcile(user)

    def sync_status(self, user: str, path: str) -> dict[str, Any]:
        canonical = "/".join(self._visible_parts(path))
        if self.sync is None:
            return {"path": canonical, "state": "error", "revision": None, "sha256": None, "error": self._sync_init_error}
        return self.sync.statuses(user).get(
            canonical,
            {"path": canonical, "state": "pending", "revision": None, "sha256": None},
        )

    def create_text_file(self, user_id: str, session_id: str, path: str, content: str) -> dict[str, Any]:
        result = super().create_text_file(user_id, session_id, path, content)
        self._attach_sync(result, user_id)
        return result

    def edit_text_file(self, user_id: str, path: str, old_text: str, new_text: str, expected_sha256: str) -> dict[str, Any]:
        result = super().edit_text_file(user_id, path, old_text, new_text, expected_sha256)
        self._attach_sync(result, user_id)
        return result

    def import_bytes(self, user: str, session: str, path: str, data: bytes) -> dict[str, Any]:
        session_id = self._safe_session_id(session)
        parts = self._write_parts(path)
        relative = "/".join((session_id, *parts))
        if not isinstance(data, bytes) or len(data) > MAX_BYTES:
            raise ValueError(f"data must be bytes no larger than {MAX_BYTES} bytes")
        UserFileSystem.write_bytes(self, user, relative, data, expected_sha256=None)
        result = {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "created": True}
        self._attach_sync(result, user)
        return result

    def _attach_sync(self, result: dict[str, Any], user: str) -> None:
        try:
            if self.sync is None:
                result["sync"] = {"path": result["path"], "state": "error", "revision": None, "sha256": result.get("sha256"), "error": self._sync_init_error}
                return
            self.sync.reconcile(user)
            result["sync"] = self.sync_status(user, result["path"])
        except Exception as exc:
            result["sync"] = {
                "path": result["path"],
                "state": "error",
                "revision": None,
                "sha256": result.get("sha256"),
                "error": _error_message(exc),
            }
