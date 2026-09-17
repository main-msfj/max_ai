"""Crash-safe, per-user three-way synchronization for workspace files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

from .artifacts import Artifact, ArtifactConflict, ArtifactStore, LocalArtifactStore
from .filesystem import UserFileSystem

MAX_FILES = 1000
MAX_BYTES = 8 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_THREAD_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class WorkspaceSync:
    def __init__(self, root: str | os.PathLike[str], store: ArtifactStore | None = None):
        self.root = Path(root).expanduser().resolve()
        self.fs = UserFileSystem(self.root)
        self.store = store if store is not None else LocalArtifactStore(self.root / ".artifacts")
        self.journal = self.root / ".workspace-sync.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS journal (store TEXT, user TEXT, path TEXT, base_hash TEXT, base_revision TEXT, state TEXT, error TEXT, PRIMARY KEY(store,user,path))")
            conn.execute("CREATE TABLE IF NOT EXISTS legacy_import (store TEXT, user TEXT, source TEXT, digest TEXT, state TEXT, PRIMARY KEY(store,user,source))")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.journal, timeout=30)) as conn:
            with conn:
                yield conn

    @contextmanager
    def _locked(self, user: str) -> Iterator[None]:
        key = (str(self.root), user)
        with _THREAD_LOCKS_GUARD:
            lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with lock:
            lock_dir = self.root / ".workspace-sync-locks"
            lock_dir.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(lock_dir / (hashlib.sha256(user.encode()).hexdigest() + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _journal_key(self) -> str:
        identity = self.store.identity
        if not isinstance(identity, str) or not identity:
            raise ValueError("invalid artifact store identity")
        return identity

    def _journal(self, user: str) -> dict[str, tuple[str | None, str | None, str, str | None]]:
        with self._db() as conn:
            rows = conn.execute("SELECT path,base_hash,base_revision,state,error FROM journal WHERE store=? AND user=?", (self._journal_key(), user)).fetchall()
        return {p: (h, rev, state, err) for p, h, rev, state, err in rows}

    def _save(self, user: str, path: str, digest: str | None, revision: str | None, state: str, error: str | None = None) -> None:
        with self._db() as conn:
            conn.execute("INSERT INTO journal VALUES(?,?,?,?,?,?,?) ON CONFLICT(store,user,path) DO UPDATE SET base_hash=excluded.base_hash,base_revision=excluded.base_revision,state=excluded.state,error=excluded.error", (self._journal_key(), user, path, digest, revision, state, error))

    def _legacy_imports(self, user: str) -> tuple[set[str], list[dict[str, Any]], bool]:
        """Copy old artifacts into legacy/ through the descriptor-safe facade."""
        key = self._journal_key()
        with self._db() as conn:
            existing = {source: (digest, state) for source, digest, state in conn.execute(
                "SELECT source,digest,state FROM legacy_import WHERE store=? AND user=?", (key, user)
            )}
        source_fs = UserFileSystem(self.root / user)
        try:
            self.fs.user_root(user)
            entries, truncated = source_fs.scan_files("artifacts", limit=MAX_FILES)
        except FileNotFoundError:
            return set(), [], False
        except Exception:
            return set(), [_item("legacy/<unavailable>", "error", None, None, "legacy artifact scan failed")], False
        blocked: set[str] = set()
        results: list[dict[str, Any]] = []
        for entry in entries:
            try:
                source = "/".join(UserFileSystem._visible_parts(entry.get("path")))
                self._safe_path(f"legacy/{source}")
                destination = f"legacy/{source}"
                record = existing.get(source)
                if record is not None:
                    if record[1] == "conflict":
                        blocked.add(destination)
                        results.append(_item(destination, "conflict", None, record[0], "legacy destination already exists; original retained"))
                        continue
                    if record[1] == "imported":
                        continue
                if entry.get("bytes", 0) > MAX_BYTES:
                    digest, state = None, "error"
                    message = "legacy file exceeds 8388608 bytes"
                else:
                    data, _ = source_fs._read_all("artifacts", UserFileSystem._visible_parts(source))
                    digest = hashlib.sha256(data).hexdigest()
                    try:
                        self.fs.write_bytes(user, destination, data, expected_sha256=None)
                        state, message = "imported", None
                    except FileExistsError:
                        state, message = "conflict", "legacy destination already exists; original retained"
                        blocked.add(destination)
                    except Exception:
                        state, message = "error", "legacy import failed"
                with self._db() as conn:
                    conn.execute("INSERT OR REPLACE INTO legacy_import VALUES(?,?,?,?,?)", (key, user, source, digest, state))
                if state == "conflict":
                    self._save(user, destination, digest, None, "conflict", message)
                    results.append(_item(destination, "conflict", None, digest, message))
                elif state == "error":
                    results.append(_item(destination, "error", None, digest, message))
            except Exception:
                results.append(_item("legacy/<invalid>", "error", None, None, "legacy import validation failed"))
        return blocked, results, truncated

    @staticmethod
    def _safe_path(path: Any) -> str:
        return "/".join(UserFileSystem._conversation_file_parts(path))

    @staticmethod
    def _metadata(artifact: Artifact) -> tuple[str, str, str, int]:
        path = WorkspaceSync._safe_path(getattr(artifact, "path", None))
        revision = getattr(artifact, "revision", None)
        digest = getattr(artifact, "sha256", None)
        size = getattr(artifact, "size", None)
        if not isinstance(revision, str) or not revision or len(revision) > 1024:
            raise ValueError("invalid revision")
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("invalid digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid size")
        return path, revision, digest, size

    def _local_files(self, user: str) -> tuple[dict[str, tuple[str, int]], dict[str, str], bool]:
        entries, truncated = self.fs.scan_files(user, limit=MAX_FILES)
        files: dict[str, tuple[str, int]] = {}
        errors: dict[str, str] = {}
        for entry in entries:
            try:
                path = self._safe_path(entry.get("path"))
                if entry.get("bytes", 0) > MAX_BYTES:
                    errors[path] = "file exceeds 8388608 bytes"
                    continue
                data = self.fs.read_snapshot(user, path)
                files[path] = (hashlib.sha256(data).hexdigest(), len(data))
            except FileNotFoundError:
                continue
            except Exception as exc:
                try:
                    errors[self._safe_path(entry.get("path"))] = _error_message(exc)
                except Exception:
                    continue
        return files, errors, truncated

    def reconcile(self, user: str) -> dict[str, Any]:
        user = UserFileSystem._safe_id(user, "user_id")
        with self._locked(user):
            try:
                blocked_legacy, legacy_items, legacy_truncated = self._legacy_imports(user)
                local, local_errors, local_truncated = self._local_files(user)
                journal = self._journal(user)
            except Exception:
                return {"items": [], "truncated": False, "state": "error", "error": "workspace synchronization state unavailable"}
            try:
                remote_list, remote_truncated = self.store.list(user, limit=MAX_FILES)
            except Exception:
                items = list(legacy_items)
                paths = set(local) | set(local_errors) | set(journal)
                for path in sorted(paths):
                    prior = journal.get(path, (None, None, "pending", None))
                    if path in blocked_legacy:
                        continue
                    changed = path in local and (prior[0] is None or local[path][0] != prior[0])
                    state = "pending" if changed else "error"
                    self._save(user, path, prior[0], prior[1], state, "artifact store unavailable")
                    items.append(_item(path, state, prior[1], prior[0], "artifact store unavailable"))
                return {
                    "items": items,
                    "truncated": bool(local_truncated or legacy_truncated),
                    "state": "error",
                    "error": "artifact store unavailable",
                }

            remote: dict[str, Artifact] = {}
            remote_errors: dict[str, str] = {}
            invalid_remote: list[str] = []
            for artifact in remote_list:
                path: str | None = None
                try:
                    path = self._safe_path(getattr(artifact, "path", None))
                    self._metadata(artifact)
                    if path in remote:
                        remote.pop(path, None)
                        remote_errors[path] = "duplicate remote artifact path"
                    else:
                        if path not in remote_errors:
                            remote[path] = artifact
                except Exception:
                    # Never echo untrusted metadata (it can contain paths or secrets).
                    if path is None:
                        invalid_remote.append("<invalid-remote>")
                    else:
                        remote_errors[path] = "invalid remote artifact metadata"
            truncated = bool(local_truncated or remote_truncated or legacy_truncated)
            paths = set(local) | set(local_errors) | set(remote) | set(journal) | set(remote_errors)
            items: list[dict[str, Any]] = list(legacy_items)
            for path in sorted(paths):
                if path in blocked_legacy:
                    continue
                if path in remote_errors:
                    base = journal.get(path, (None, None, "", None))
                    self._save(user, path, base[0], base[1], "error", remote_errors[path])
                    items.append(_item(path, "error", base[1], base[0], remote_errors[path]))
                    continue
                if path in local_errors:
                    prior = journal.get(path, (None, None, "error", None))
                    self._save(user, path, prior[0], prior[1], "error", local_errors[path])
                    items.append(_item(path, "error", prior[1], prior[0], local_errors[path]))
                    continue
                try:
                    items.append(self._reconcile_path(user, path, local, remote, journal, truncated))
                except ArtifactConflict:
                    base = journal.get(path, (None, None, "", None))
                    self._save(user, path, base[0], base[1], "conflict", "artifact revision changed during synchronization")
                    items.append(_item(path, "conflict", base[1], base[0], "artifact revision changed during synchronization"))
                except Exception as exc:
                    base = journal.get(path, (None, None, "", None))
                    message = _error_message(exc)
                    self._save(user, path, base[0], base[1], "error", message)
                    items.append(_item(path, "error", base[1], base[0], message))
            items.extend(_item("<invalid-remote>", "error", None, None, "invalid remote artifact metadata") for _ in invalid_remote)
            return {"items": items, "truncated": truncated}

    def _reconcile_path(self, user: str, path: str, local: dict[str, tuple[str, int]], remote: dict[str, Artifact], journal: dict[str, Any], truncated: bool) -> dict[str, Any]:
        local_info = local.get(path)
        local_hash = local_info[0] if local_info else None
        artifact = remote.get(path)
        remote_meta = self._metadata(artifact) if artifact is not None else None
        remote_revision = remote_meta[1] if remote_meta else None
        remote_hash = remote_meta[2] if remote_meta else None
        base_hash, base_revision, _, _ = journal.get(path, (None, None, "", None))

        def saved(state: str, digest: str | None, revision: str | None, error: str | None = None) -> dict[str, Any]:
            self._save(user, path, digest, revision, state, error)
            return _item(path, state, revision, digest, error)

        if truncated and artifact is None:
            return saved("pending", base_hash, base_revision, "scan truncated; absence cannot be inferred")
        if truncated and local_info is None:
            return saved("pending", base_hash, base_revision, "scan truncated; absence cannot be inferred")
        if artifact is None and local_info is None:
            if base_hash is not None:
                return saved("conflict", base_hash, base_revision, "remote artifact is missing; deletion was not propagated")
            return saved("synced", None, None)
        if artifact is None:
            if base_hash is not None:
                return saved("conflict", base_hash, base_revision, "remote artifact is missing; deletion was not propagated")
            data = self._read_local_checked(user, path, local_hash)
            self._save(user, path, base_hash, base_revision, "pending")
            uploaded = self.store.put(user, path, data, expected_revision=None)
            return self._saved_from_put(user, path, uploaded, local_hash, len(data))

        assert remote_meta is not None
        if remote_meta[3] > MAX_BYTES:
            return saved("error", base_hash, base_revision, "remote file exceeds 8388608 bytes")
        remote_data: bytes | None = None
        # The persisted baseline lets us avoid a download while the opaque
        # remote revision is unchanged. A new revision must be read and hashed;
        # provider metadata digests can lag the actual blob contents.
        if base_hash is not None and remote_revision == base_revision:
            remote_hash = base_hash
        else:
            remote_data = self._read_remote_checked(user, path, remote_meta)
            remote_hash = hashlib.sha256(remote_data).hexdigest()

        if local_info is None:
            self._save(user, path, base_hash, base_revision, "pending")
            data = remote_data if remote_data is not None else self._read_remote_checked(user, path, remote_meta)
            self.fs.write_bytes(user, path, data, expected_sha256=None)
            return saved("synced", remote_hash, remote_revision)
        if local_hash == remote_hash:
            return saved("synced", local_hash, remote_revision)
        if base_hash is None:
            return saved("conflict", None, None, "local and remote files were independently created with different content")

        local_changed = local_hash != base_hash
        remote_changed = remote_hash != base_hash
        if local_changed and not remote_changed:
            data = self._read_local_checked(user, path, local_hash)
            self._save(user, path, base_hash, base_revision, "pending")
            uploaded = self.store.put(user, path, data, expected_revision=remote_revision)
            return self._saved_from_put(user, path, uploaded, local_hash, len(data))
        if not local_changed and remote_changed:
            self._save(user, path, base_hash, base_revision, "pending")
            data = remote_data if remote_data is not None else self._read_remote_checked(user, path, remote_meta)
            self.fs.write_bytes(user, path, data, expected_sha256=local_hash)
            return saved("synced", remote_hash, remote_revision)
        return saved("conflict", base_hash, base_revision, "both local and remote files changed")

    def _read_local_checked(self, user: str, path: str, expected_hash: str) -> bytes:
        data = self.fs.read_snapshot(user, path)
        if len(data) > MAX_BYTES:
            raise ValueError("local file exceeds 8388608 bytes")
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("local file changed during synchronization")
        return data

    def _read_remote_checked(self, user: str, path: str, expected: tuple[str, str, str, int]) -> bytes:
        artifact, data = self.store.get(user, path)
        actual = self._metadata(artifact)
        if actual[0] != expected[0] or actual[1] != expected[1] or actual[3] != expected[3] or len(data) != expected[3]:
            raise ValueError("remote content failed integrity verification")
        return data

    def _saved_from_put(self, user: str, path: str, artifact: Artifact, digest: str, size: int) -> dict[str, Any]:
        meta = self._metadata(artifact)
        if meta[0] != path or meta[2] != digest or meta[3] != size:
            self._save(user, path, digest, None, "error", "artifact store returned invalid metadata")
            return _item(path, "error", None, digest, "artifact store returned invalid metadata")
        self._save(user, path, digest, meta[1], "synced")
        return _item(path, "synced", meta[1], digest)

    def statuses(self, user: str) -> dict[str, dict[str, Any]]:
        user = UserFileSystem._safe_id(user, "user_id")
        result: dict[str, dict[str, Any]] = {}
        for path, (digest, revision, state, error) in self._journal(user).items():
            result[path] = _item(path, state, revision, digest, error)
        return result


def _item(path: str, state: str, revision: str | None, digest: str | None, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "state": state, "revision": revision, "sha256": digest}
    if error:
        result["error"] = error
    return result


def _error_message(exc: Exception) -> str:
    if isinstance(exc, ArtifactConflict):
        return "artifact revision changed during synchronization"
    if isinstance(exc, FileNotFoundError):
        return "workspace file changed during synchronization"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "artifact store unavailable"
    if isinstance(exc, ValueError):
        return "workspace or artifact validation failed"
    return "synchronization failed"
