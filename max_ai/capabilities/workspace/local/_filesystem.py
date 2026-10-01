"""User-scoped, POSIX filesystem access for model-facing file tools.

Tool writes are serialized by process-local path locks. They do not coordinate
with another process that edits the same workspace concurrently.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import stat
import threading
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_RESERVED_ROOT_DIRS = {"tools", "skills", "artifacts", "workspace"}
# Hidden from root listings (ListDirectory/FindFiles at the user root).
# workspace stays visible so FindFiles/SearchFile keep descending into it.
_HIDDEN_ROOT_DIRS = _RESERVED_ROOT_DIRS - {"skills", "workspace"}
# Blocked from explicit path resolution too. skills/workspace stay
# reachable by name; tools/artifacts are fully off-limits.
_BLOCKED_PATH_ROOTS = _RESERVED_ROOT_DIRS - {"skills", "workspace"}
_INTERNAL_TEMP_PREFIX = ".maxai-"
_MAX_PATH_BYTES = 4096
_MAX_SEGMENT_BYTES = 255
_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_READ_BYTES = 8 * 1024 * 1024
_MAX_WRITE_BYTES = 1024 * 1024
# Formats that text tools can only corrupt: they must be produced by a program.
_BINARY_SUFFIXES = frozenset({
    ".xlsx", ".xlsm", ".xls", ".docx", ".doc", ".pptx", ".ppt", ".pdf", ".zip",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp3", ".mp4", ".sqlite", ".db",
})
_MAX_BINARY_WRITE_BYTES = 8 * 1024 * 1024
_MAX_LIST_LIMIT = 500
_MAX_SCAN_FILES = 1000
_MAX_SCAN_ENTRIES = 10000
_MAX_SCAN_DEPTH = 64

# A file as the model last saw it: (mtime_ns, size). Writes compare it with
# the file on disk, so nothing the model didn't read gets overwritten.
Stamp = tuple[int, int]

_LOCKS: dict[tuple[str, str, str], threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _path_lock(key: tuple[str, str, str]) -> threading.RLock:
    """Perform the internal ``path lock`` operation.

Parameters
----------
key : tuple[str, str, str]
    Value supplied for ``key``."""
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def stamp_of(info: os.stat_result) -> Stamp:
    """The ``Stamp`` of a file from its ``stat``."""
    return (info.st_mtime_ns, info.st_size)


class UserFileSystem:
    """Access ``root/user/workspace`` with user-wide read scope."""

    def __init__(self, root: str | os.PathLike[str]):
        """Initialize ``UserFileSystem``.

Parameters
----------
root : str | os.PathLike[str]
    Value supplied for ``root``."""
        if (
            os.name != "posix"
            or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.mkdir not in os.supports_dir_fd
        ):
            raise OSError("UserFileSystem requires POSIX dir-fd and no-follow support")
        self.root = Path(root).expanduser().resolve()

    def user_root(self, user_id: str) -> Path:
        """Validate and securely materialize one user's root directory."""
        user = self._safe_id(user_id, "user_id")
        user_fd = self._open_user_fd(user, create=True)
        os.close(user_fd)
        return self.root / user

    def workspace_root(self, user_id: str) -> Path:
        """Validate and securely materialize the user's single workspace."""
        user = self._safe_id(user_id, "user_id")
        user_fd = self._open_user_fd(user, create=True)
        try:
            workspace_fd = self._open_dir_at(user_fd, "workspace", create=True)
            os.close(workspace_fd)
        finally:
            os.close(user_fd)
        return self.root / user / "workspace"

    def list_files(
        self, user_id: str, path: str = "", limit: int = 200
    ) -> dict[str, Any]:
        """List one directory, returning user-relative file references."""
        effective_limit = self._bounded_limit(limit, _MAX_LIST_LIMIT)
        parts = self._visible_parts(path, allow_empty=True)
        display_path = "/".join(parts) or "."
        try:
            if not parts:
                directory_fd = self._open_user_fd(user_id)
                is_user_root = True
            else:
                parent_fd = self._open_parent_fd(user_id, parts)
                try:
                    leaf = parts[-1]
                    info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("symlinks are not accessible")
                    if stat.S_ISREG(info.st_mode):
                        self._require_single_link(info)
                        return {
                            "path": display_path,
                            "items": [
                                {
                                    "path": display_path,
                                    "type": "file",
                                    "bytes": info.st_size,
                                }
                            ],
                            "truncated": False,
                        }
                    if not stat.S_ISDIR(info.st_mode):
                        raise ValueError(
                            "only regular files and directories are accessible"
                        )
                    if leaf.startswith("."):
                        raise ValueError("dot directories are not accessible")
                    directory_fd = self._open_dir_at(parent_fd, leaf)
                finally:
                    os.close(parent_fd)
                is_user_root = False
        except FileNotFoundError:
            return {"path": display_path, "items": [], "truncated": False}

        try:
            items: list[dict[str, Any]] = []
            scanned = 0
            truncated = False
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if scanned >= _MAX_SCAN_ENTRIES:
                        truncated = True
                        break
                    scanned += 1
                    name = entry.name
                    if (is_user_root and name in _HIDDEN_ROOT_DIRS) or name.startswith(
                        _INTERNAL_TEMP_PREFIX
                    ):
                        continue
                    try:
                        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        if name.startswith("."):
                            continue
                        item_type = "directory"
                        size: int | None = None
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        item_type = "file"
                        size = info.st_size
                    else:
                        continue
                    items.append(
                        {
                            "path": "/".join((*parts, name)),
                            "type": item_type,
                            "bytes": size,
                        }
                    )
            items.sort(key=lambda item: item["path"])
            truncated = truncated or len(items) > effective_limit
            return {
                "path": display_path,
                "items": items[:effective_limit],
                "truncated": truncated,
            }
        finally:
            os.close(directory_fd)

    def scan_files(
        self, user_id: str, path: str = "", limit: int = _MAX_SCAN_FILES
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return a bounded recursive file scan for find and search tools."""
        effective_limit = self._bounded_limit(limit, _MAX_SCAN_FILES)
        parts = self._visible_parts(path, allow_empty=True)
        files: list[dict[str, Any]] = []
        truncated = False

        try:
            if not parts:
                start_fd = self._open_user_fd(user_id)
                start_is_user_root = True
                base_parts: tuple[str, ...] = ()
            else:
                parent_fd = self._open_parent_fd(user_id, parts)
                try:
                    leaf = parts[-1]
                    info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("symlinks are not accessible")
                    if stat.S_ISREG(info.st_mode):
                        self._require_single_link(info)
                        return (
                            [{"path": "/".join(parts), "bytes": info.st_size}],
                            False,
                        )
                    if not stat.S_ISDIR(info.st_mode):
                        raise ValueError(
                            "only regular files and directories are accessible"
                        )
                    if leaf.startswith("."):
                        raise ValueError("dot directories are not accessible")
                    start_fd = self._open_dir_at(parent_fd, leaf)
                finally:
                    os.close(parent_fd)
                start_is_user_root = False
                base_parts = parts
        except FileNotFoundError:
            return [], False

        entries_seen = 0

        def walk(
            directory_fd: int, base: tuple[str, ...], depth: int, user_root: bool
        ) -> None:
            """Perform the ``walk`` operation for ``UserFileSystem``.

Parameters
----------
directory_fd : int
    Value supplied for ``directory_fd``.
base : tuple[str, ...]
    Value supplied for ``base``.
depth : int
    Value supplied for ``depth``.
user_root : bool
    Value supplied for ``user_root``."""
            nonlocal entries_seen, truncated
            if depth > _MAX_SCAN_DEPTH:
                truncated = True
                return
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if entries_seen >= _MAX_SCAN_ENTRIES:
                        truncated = True
                        return
                    entries_seen += 1
                    name = entry.name
                    if (user_root and name in _HIDDEN_ROOT_DIRS) or name.startswith(
                        _INTERNAL_TEMP_PREFIX
                    ):
                        continue
                    try:
                        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        continue
                    child_parts = (*base, name)
                    if stat.S_ISDIR(info.st_mode):
                        if name.startswith("."):
                            continue
                        try:
                            child_fd = self._open_dir_at(directory_fd, name)
                        except FileNotFoundError:
                            continue
                        except ValueError:
                            continue
                        try:
                            walk(child_fd, child_parts, depth + 1, False)
                        finally:
                            os.close(child_fd)
                        if truncated:
                            return
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        files.append(
                            {"path": "/".join(child_parts), "bytes": info.st_size}
                        )
                        if len(files) > effective_limit:
                            truncated = True
                            return

        try:
            walk(start_fd, base_parts, 0, start_is_user_root)
        finally:
            os.close(start_fd)
        files.sort(key=lambda item: item["path"])
        return files[:effective_limit], truncated

    def read_bytes(self, user_id: str, path: str, max_bytes: int) -> bytes:
        """Read at most ``max_bytes`` from one regular, singly-linked file."""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("max_bytes must be an integer")
        if max_bytes < 0 or max_bytes > _MAX_READ_BYTES:
            raise ValueError(f"max_bytes must be between 0 and {_MAX_READ_BYTES}")
        parts = self._visible_parts(path)
        parent_fd, file_fd, _ = self._open_regular_file(user_id, parts)
        try:
            result = bytearray()
            while len(result) < max_bytes:
                chunk = os.read(file_fd, min(65536, max_bytes - len(result)))
                if not chunk:
                    break
                result.extend(chunk)
            return bytes(result)
        finally:
            os.close(file_fd)
            os.close(parent_fd)

    def read_snapshot(self, user_id: str, path: str) -> bytes:
        """Read a complete file from a visible conversation path, up to 8 MiB."""
        parts = self._workspace_file_parts(path)
        data, _ = self._read_all(user_id, parts)
        return data

    def write_bytes(
        self, user_id: str, path: str, data: bytes, expected: Stamp | None = None
    ) -> dict[str, Any]:
        """Atomically create a file, or replace it when ``expected`` matches."""
        if not isinstance(data, bytes):
            raise ValueError("data must be bytes")
        if len(data) > _MAX_BINARY_WRITE_BYTES:
            raise ValueError(f"data exceeds {_MAX_BINARY_WRITE_BYTES} bytes")
        return self._write(user_id, self._workspace_file_parts(path), data, expected)

    def read_text(self, user_id: str, path: str) -> dict[str, Any]:
        """A whole file (up to 8 MiB) with its stamp; ``text`` is None for binary."""
        parts = self._visible_parts(path)
        data, info = self._read_all(user_id, parts)
        try:
            text: str | None = data.decode("utf-8")
            if "\0" in text:
                text = None
        except UnicodeDecodeError:
            text = None
        return {
            "path": "/".join(parts),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "stamp": stamp_of(info),
            "text": text,
        }

    def write_text_file(
        self, user_id: str, path: str, content: str, expected: Stamp | None = None
    ) -> dict[str, Any]:
        """Create a text file, or replace it when ``expected`` matches."""
        parts = self._workspace_file_parts(path)
        if Path(parts[-1]).suffix.lower() in _BINARY_SUFFIXES:
            # Text written into e.g. .xlsx is a broken file, not a spreadsheet.
            raise ValueError(
                f"{parts[-1]} is a binary format: create it by running code "
                "(for example a Python script with bash), not with WriteFile"
            )
        return self._write(user_id, parts, self._text_bytes(content), expected)

    def _write(
        self, user_id: str, parts: tuple[str, ...], data: bytes, expected: Stamp | None
    ) -> dict[str, Any]:
        """Create ``parts`` (``expected`` None) or replace it if its stamp matches."""
        user = self._safe_id(user_id, "user_id")
        display_path = "/".join(parts)
        with _path_lock((str(self.root), user, display_path)):
            parent_fd: int | None = None
            temp_name: str | None = None
            try:
                if expected is None:
                    parent_fd = self._open_parent_fd(user, parts, create=True)
                    temp_name = self._write_temp(parent_fd, data)
                    try:
                        os.link(
                            temp_name,
                            parts[-1],
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        raise FileExistsError(
                            f"{display_path} already exists: read it before overwriting it"
                        ) from None
                    os.unlink(temp_name, dir_fd=parent_fd)
                    temp_name = None
                else:
                    parent_fd = self._open_parent_fd(user, parts)
                    try:
                        current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"{display_path} no longer exists") from None
                    self._require_regular_single_link(current)
                    self._check_stamp(parts, current, expected)
                    temp_name = self._write_temp(parent_fd, data)
                    os.replace(
                        temp_name,
                        parts[-1],
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    temp_name = None
                os.fsync(parent_fd)
                written = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            finally:
                if parent_fd is not None:
                    if temp_name is not None:
                        self._unlink_if_present(parent_fd, temp_name)
                    os.close(parent_fd)
        return {
            "path": display_path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "created": expected is None,
            "stamp": stamp_of(written),
        }

    def edit_text_file(
        self,
        user_id: str,
        path: str,
        old_text: str,
        new_text: str,
        expected: Stamp,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Replace ``old_text`` (once, or everywhere with ``replace_all``) if the
        file still matches ``expected``."""
        parts = self._visible_parts(path)
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("old_string must not be empty")
        if old_text == new_text:
            raise ValueError("old_string and new_string are the same: nothing to change")
        self._text_bytes(new_text)
        user = self._safe_id(user_id, "user_id")
        lock = _path_lock((str(self.root), user, "/".join(parts)))
        with lock:
            original, opened_stat = self._read_all(user, parts)
            self._check_stamp(parts, opened_stat, expected)
            try:
                text = original.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("binary files cannot be edited as text") from None
            if "\0" in text:
                raise ValueError("binary files cannot be edited as text")
            count = text.count(old_text)
            if count == 0:
                raise ValueError(
                    "old_string was not found. Copy it exactly from ReadFile, "
                    "without the line numbers, and keep its whitespace."
                )
            if count > 1 and not replace_all:
                raise ValueError(
                    f"old_string appears {count} times. Add surrounding lines to make "
                    "it unique, or set replace_all to change every one."
                )
            updated = text.replace(old_text, new_text, -1 if replace_all else 1).encode("utf-8")
            if len(updated) > _MAX_WRITE_BYTES:
                raise ValueError(f"edited file exceeds {_MAX_WRITE_BYTES} bytes")

            parent_fd = self._open_parent_fd(user, parts)
            temp_name: str | None = None
            try:
                leaf = parts[-1]
                current_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                self._require_regular_single_link(current_stat)
                if (
                    current_stat.st_dev != opened_stat.st_dev
                    or current_stat.st_ino != opened_stat.st_ino
                    or current_stat.st_size != opened_stat.st_size
                    or current_stat.st_mtime_ns != opened_stat.st_mtime_ns
                ):
                    raise ValueError("file changed while it was being edited")
                temp_name = self._write_temp(parent_fd, updated)
                os.replace(
                    temp_name,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                temp_name = None
                os.fsync(parent_fd)
                written = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            finally:
                if temp_name is not None:
                    self._unlink_if_present(parent_fd, temp_name)
                os.close(parent_fd)
        return {
            "path": "/".join(parts),
            "bytes": len(updated),
            "sha256": hashlib.sha256(updated).hexdigest(),
            "replacements": count if replace_all else 1,
            "stamp": stamp_of(written),
        }

    def delete_file(self, user_id: str, path: str, expected: Stamp) -> dict[str, Any]:
        """Delete one regular file if it still matches ``expected``."""
        parts = self._workspace_file_parts(path)
        user = self._safe_id(user_id, "user_id")
        display_path = "/".join(parts)
        lock = _path_lock((str(self.root), user, display_path))
        with lock:
            original, opened_stat = self._read_all(user, parts)
            self._check_stamp(parts, opened_stat, expected)
            digest = hashlib.sha256(original).hexdigest()
            parent_fd = self._open_parent_fd(user, parts)
            try:
                current_stat = os.stat(
                    parts[-1], dir_fd=parent_fd, follow_symlinks=False
                )
                self._require_regular_single_link(current_stat)
                if (
                    current_stat.st_dev != opened_stat.st_dev
                    or current_stat.st_ino != opened_stat.st_ino
                    or current_stat.st_size != opened_stat.st_size
                    or current_stat.st_mtime_ns != opened_stat.st_mtime_ns
                ):
                    raise ValueError("file changed while it was being deleted")
                os.unlink(parts[-1], dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return {"path": display_path, "deleted": True, "sha256": digest}

    @staticmethod
    def _safe_id(value: str, label: str) -> str:
        """Perform the internal ``safe id`` operation for ``UserFileSystem``.

Parameters
----------
value : str
    Value supplied for ``value``.
label : str
    Value supplied for ``label``."""
        if not isinstance(value, str) or not _ID_RE.fullmatch(value):
            raise ValueError(f"{label} must be a safe single path segment")
        return value

    @classmethod
    def _safe_session_id(cls, value: str) -> str:
        """Perform the internal ``safe session id`` operation for ``UserFileSystem``.

Parameters
----------
value : str
    Value supplied for ``value``."""
        session = cls._safe_id(value, "session_id")
        if session in _RESERVED_ROOT_DIRS:
            raise ValueError("session_id must not be a reserved directory name")
        return session

    @staticmethod
    def _bounded_limit(limit: int, maximum: int) -> int:
        """Perform the internal ``bounded limit`` operation for ``UserFileSystem``.

Parameters
----------
limit : int
    Value supplied for ``limit``.
maximum : int
    Value supplied for ``maximum``."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        return min(limit, maximum)

    @staticmethod
    def _parts(path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        """Perform the internal ``parts`` operation for ``UserFileSystem``.

Parameters
----------
path : str
    Value supplied for ``path``.
allow_empty : bool
    Value supplied for ``allow_empty``."""
        if not isinstance(path, str):
            raise ValueError("path must be a relative string")
        if len(path.encode("utf-8", errors="strict")) > _MAX_PATH_BYTES:
            raise ValueError("path is too long")
        if "\0" in path or "\\" in path or path.startswith("/"):
            raise ValueError("path must be relative and contain no NUL or backslash")
        if path == "":
            if allow_empty:
                return ()
            raise ValueError("path must name a file")
        parts = tuple(path.split("/"))
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("path must not contain empty, '.' or '..' segments")
        if any(len(part.encode("utf-8")) > _MAX_SEGMENT_BYTES for part in parts):
            raise ValueError("path contains a segment that is too long")
        return parts

    @classmethod
    def _visible_parts(cls, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        """Perform the internal ``visible parts`` operation for ``UserFileSystem``.

Parameters
----------
path : str
    Value supplied for ``path``.
allow_empty : bool
    Value supplied for ``allow_empty``."""
        parts = cls._parts(path, allow_empty=allow_empty)
        if parts and parts[0] in _BLOCKED_PATH_ROOTS:
            raise ValueError("tools and artifacts are outside the user file scope")
        if any(part.startswith(_INTERNAL_TEMP_PREFIX) for part in parts):
            raise ValueError("internal temporary files are not accessible")
        if any(part.startswith(".") for part in parts[:-1]):
            raise ValueError("dot directories are not accessible")
        return parts

    @classmethod
    def _write_parts(cls, path: str) -> tuple[str, ...]:
        """Perform the internal ``write parts`` operation for ``UserFileSystem``.

Parameters
----------
path : str
    Value supplied for ``path``."""
        parts = cls._visible_parts(path)
        if any(part.startswith(".") for part in parts):
            raise ValueError("dot paths are not writable")
        return parts

    @classmethod
    def _workspace_file_parts(cls, path: str) -> tuple[str, ...]:
        """Perform the internal ``workspace file parts`` operation for ``UserFileSystem``.

Parameters
----------
path : str
    Value supplied for ``path``."""
        parts = cls._write_parts(path)
        if len(parts) < 2 or parts[0] != "workspace":
            raise ValueError("path must be a workspace/<file> path")
        return parts

    @staticmethod
    def _text_bytes(content: str) -> bytes:
        """Perform the internal ``text bytes`` operation for ``UserFileSystem``.

Parameters
----------
content : str
    Value supplied for ``content``."""
        if not isinstance(content, str):
            raise ValueError("content must be text")
        if "\0" in content:
            raise ValueError("NUL bytes are not allowed in text files")
        data = content.encode("utf-8")
        if len(data) > _MAX_WRITE_BYTES:
            raise ValueError(f"content exceeds {_MAX_WRITE_BYTES} bytes")
        return data

    def _open_root_fd(self, *, create: bool = False) -> int:
        """Perform the internal ``open root fd`` operation for ``UserFileSystem``.

Parameters
----------
create : bool
    Value supplied for ``create``."""
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            return os.open(self.root, flags)
        except OSError as exc:
            self._raise_safe_path_error(exc)
            raise

    @staticmethod
    def _open_dir_at(parent_fd: int, name: str, *, create: bool = False) -> int:
        """Perform the internal ``open dir at`` operation for ``UserFileSystem``.

Parameters
----------
parent_fd : int
    Value supplied for ``parent_fd``.
name : str
    Value supplied for ``name``.
create : bool
    Value supplied for ``create``."""
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            return os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            try:
                return os.open(name, flags, dir_fd=parent_fd)
            except OSError as exc:
                UserFileSystem._raise_safe_path_error(exc)
                raise
        except OSError as exc:
            UserFileSystem._raise_safe_path_error(exc)
            raise

    def _open_user_fd(self, user_id: str, *, create: bool = False) -> int:
        """Perform the internal ``open user fd`` operation for ``UserFileSystem``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
create : bool
    Value supplied for ``create``."""
        user = self._safe_id(user_id, "user_id")
        root_fd = self._open_root_fd(create=create)
        try:
            return self._open_dir_at(root_fd, user, create=create)
        finally:
            os.close(root_fd)

    def _open_parent_fd(
        self, user_id: str, parts: tuple[str, ...], *, create: bool = False
    ) -> int:
        """Perform the internal ``open parent fd`` operation for ``UserFileSystem``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
parts : tuple[str, ...]
    Value supplied for ``parts``.
create : bool
    Value supplied for ``create``."""
        if not parts:
            raise ValueError("path must name a file")
        fd = self._open_user_fd(user_id, create=create)
        try:
            for part in parts[:-1]:
                if part.startswith("."):
                    raise ValueError("dot directories are not accessible")
                child_fd = self._open_dir_at(fd, part, create=create)
                os.close(fd)
                fd = child_fd
            return fd
        except Exception:
            os.close(fd)
            raise

    def _open_regular_file(
        self, user_id: str, parts: tuple[str, ...]
    ) -> tuple[int, int, os.stat_result]:
        """Perform the internal ``open regular file`` operation for ``UserFileSystem``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
parts : tuple[str, ...]
    Value supplied for ``parts``."""
        parent_fd = self._open_parent_fd(user_id, parts)
        try:
            leaf = parts[-1]
            info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            self._require_regular_single_link(info)
            flags = (
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_NONBLOCK
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                file_fd = os.open(leaf, flags, dir_fd=parent_fd)
            except OSError as exc:
                self._raise_safe_path_error(exc)
                raise
            opened_info = os.fstat(file_fd)
            try:
                self._require_regular_single_link(opened_info)
            except Exception:
                os.close(file_fd)
                raise
            return parent_fd, file_fd, opened_info
        except Exception:
            os.close(parent_fd)
            raise

    def _read_all(
        self, user_id: str, parts: tuple[str, ...]
    ) -> tuple[bytes, os.stat_result]:
        """Perform the internal ``read all`` operation for ``UserFileSystem``.

Parameters
----------
user_id : str
    Value supplied for ``user_id``.
parts : tuple[str, ...]
    Value supplied for ``parts``."""
        parent_fd, file_fd, info = self._open_regular_file(user_id, parts)
        try:
            data = bytearray()
            while True:
                chunk = os.read(file_fd, min(65536, _MAX_FILE_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > _MAX_FILE_BYTES:
                    raise ValueError(f"file exceeds {_MAX_FILE_BYTES} bytes")
            return bytes(data), info
        finally:
            os.close(file_fd)
            os.close(parent_fd)

    @staticmethod
    def _require_single_link(info: os.stat_result) -> None:
        """Perform the internal ``require single link`` operation for ``UserFileSystem``.

Parameters
----------
info : os.stat_result
    Value supplied for ``info``."""
        if info.st_nlink != 1:
            raise ValueError("hard-linked files are not accessible")

    @classmethod
    def _require_regular_single_link(cls, info: os.stat_result) -> None:
        """Perform the internal ``require regular single link`` operation for ``UserFileSystem``.

Parameters
----------
info : os.stat_result
    Value supplied for ``info``."""
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlinks are not accessible")
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("only regular files are accessible")
        cls._require_single_link(info)

    @staticmethod
    def _check_stamp(parts: tuple[str, ...], info: os.stat_result, expected: Stamp) -> None:
        """Refuse when the file on disk is not the one the model read."""
        if stamp_of(info) != tuple(expected):
            raise ValueError(f"{'/'.join(parts)} changed since it was read: read it again")

    @staticmethod
    def _raise_safe_path_error(exc: OSError) -> None:
        """Perform the internal ``raise safe path error`` operation for ``UserFileSystem``.

Parameters
----------
exc : OSError
    Value supplied for ``exc``."""
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
            raise ValueError(
                "symlinks and non-directory path components are not accessible"
            ) from None

    @staticmethod
    def _write_all(file_fd: int, data: bytes) -> None:
        """Perform the internal ``write all`` operation for ``UserFileSystem``.

Parameters
----------
file_fd : int
    Value supplied for ``file_fd``.
data : bytes
    Value supplied for ``data``."""
        view = memoryview(data)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("file write made no progress")
            view = view[written:]

    def _write_temp(self, parent_fd: int, data: bytes) -> str:
        """Perform the internal ``write temp`` operation for ``UserFileSystem``.

Parameters
----------
parent_fd : int
    Value supplied for ``parent_fd``.
data : bytes
    Value supplied for ``data``."""
        name = f".maxai-{secrets.token_hex(12)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
        try:
            self._write_all(file_fd, data)
            os.fsync(file_fd)
        except Exception:
            os.close(file_fd)
            self._unlink_if_present(parent_fd, name)
            raise
        os.close(file_fd)
        return name

    @staticmethod
    def _unlink_if_present(parent_fd: int, name: str) -> None:
        """Perform the internal ``unlink if present`` operation for ``UserFileSystem``.

Parameters
----------
parent_fd : int
    Value supplied for ``parent_fd``.
name : str
    Value supplied for ``name``."""
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
