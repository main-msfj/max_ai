"""Bounded, symlink-safe synchronization of regular files in a workspace.

Snapshots exclude file modes and empty directories. Applying a snapshot keeps
directories and does not claim transactionality; directory creation may occur
before a later file conflict is detected.

Host-side, this is a plain import (``ModalExecutor.sync()`` calls
``snapshot``/``apply_snapshot`` directly). Sandbox-side, it's invoked as
``python -m max_ai.capabilities.executor.modal.sync snapshot|apply <root>``
(see ``_executor.py``) — the dotted path is a literal string there, so
moving or renaming this file requires updating that string too. Only Modal
uses this: Local/Docker already share a filesystem with the host via bind
mount, so they never need an explicit snapshot/apply round-trip.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import stat
import sys
from pathlib import Path
from typing import Iterator
from uuid import uuid4

MAX_BYTES = 8 * 1024 * 1024
MAX_FILES = 10_000
MAX_JSON_BYTES = 24 * 1024 * 1024
_OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_OPEN_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_CHUNK = 1024 * 1024


def _checked_limit(max_bytes: int) -> int:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    return max_bytes


def _open_root(root: Path) -> int:
    if not isinstance(root, Path):
        root = Path(root)
    parts = root.parts
    if not parts:
        return os.open(".", _OPEN_DIR)
    fd = os.open(os.sep if root.is_absolute() else ".", _OPEN_DIR)
    try:
        for part in parts[1:] if root.is_absolute() else parts:
            if part in ("", ".", ".."):
                raise ValueError("root contains a non-canonical component")
            nxt = os.open(part, _OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except Exception:
        os.close(fd)
        raise


def _validate_rel(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("invalid relative path")
    if path.startswith("/") or path in (".", ".."):
        raise ValueError("invalid relative path")
    parts = path.split("/")
    if any(not p or p in (".", "..") for p in parts):
        raise ValueError("invalid relative path")
    if "/".join(parts) != path:
        raise ValueError("non-canonical relative path")
    return path


def _read_file(dirfd: int, name: str, remaining: int) -> bytes:
    try:
        fd = os.open(name, _OPEN_FILE, dir_fd=dirfd)
    except OSError as exc:
        raise ValueError(f"cannot open regular file {name!r}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"special file or symlink rejected: {name}")
        if st.st_size > remaining:
            raise ValueError("snapshot exceeds max_bytes")
        chunks = []
        total = 0
        while True:
            data = os.read(fd, min(_CHUNK, remaining - total + 1))
            if not data:
                break
            total += len(data)
            if total > remaining:
                raise ValueError("snapshot exceeds max_bytes")
            chunks.append(data)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _walk(fd: int, prefix: str = "", budget: list[int] | None = None) -> Iterator[tuple[str, bytes]]:
    if budget is None:
        budget = [MAX_BYTES]
    for name in os.listdir(fd):
        _validate_rel(name)
        rel = f"{prefix}/{name}" if prefix else name
        st = os.lstat(name, dir_fd=fd)
        if stat.S_ISDIR(st.st_mode):
            child = os.open(name, _OPEN_DIR, dir_fd=fd)
            try:
                yield from _walk(child, rel, budget)
            finally:
                os.close(child)
        elif stat.S_ISREG(st.st_mode):
            data = _read_file(fd, name, budget[0])
            budget[0] -= len(data)
            yield rel, data
        else:
            raise ValueError(f"symlink or special file rejected: {rel}")


def _snapshot_fd(fd: int, max_bytes: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel, data in _walk(fd, budget=[max_bytes]):
        if len(result) >= MAX_FILES:
            raise ValueError("snapshot exceeds max_files")
        result[rel] = base64.b64encode(data).decode("ascii")
    return result


def snapshot(root: Path, max_bytes: int = MAX_BYTES) -> dict[str, str]:
    """Return regular files below an existing root as relative base64 content."""
    max_bytes = _checked_limit(max_bytes)
    fd = _open_root(root)
    try:
        return _snapshot_fd(fd, max_bytes)
    finally:
        os.close(fd)


def _decode_map(value: object, label: str, max_bytes: int) -> dict[str, bytes]:
    if not isinstance(value, dict) or len(value) > MAX_FILES:
        raise ValueError(f"{label} must be an object with at most {MAX_FILES} files")
    result: dict[str, bytes] = {}
    total = 0
    for path, encoded in value.items():
        _validate_rel(path)
        if not isinstance(encoded, str):
            raise ValueError(f"{label} values must be base64 strings")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError(f"invalid base64 in {label}: {path}") from exc
        if base64.b64encode(data).decode("ascii") != encoded:
            raise ValueError(f"non-canonical base64 in {label}: {path}")
        total += len(data)
        if total > max_bytes:
            raise ValueError(f"{label} exceeds max_bytes")
        result[path] = data
    return result


def _file_bytes(dirfd: int, name: str, limit: int = MAX_BYTES) -> bytes | None:
    try:
        fd = os.open(name, _OPEN_FILE, dir_fd=dirfd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"cannot inspect {name!r}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise ValueError(f"target is not a bounded regular file: {name}")
        chunks = []
        total = 0
        while True:
            data = os.read(fd, min(_CHUNK, limit - total + 1))
            if not data:
                return b"".join(chunks)
            total += len(data)
            if total > limit:
                raise ValueError(f"target exceeds max_bytes: {name}")
            chunks.append(data)
    finally:
        os.close(fd)


def _open_parent(rootfd: int, rel: str, create: bool) -> tuple[int, str]:
    parts = rel.split("/")
    fd = os.dup(rootfd)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, _OPEN_DIR, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=fd)
                child = os.open(part, _OPEN_DIR, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except Exception:
        os.close(fd)
        raise


def apply_snapshot(root: Path, desired: dict[str, str], expected: dict[str, str], max_bytes: int = MAX_BYTES) -> None:
    """Apply a snapshot after an exact pre-mutation check of the current snapshot."""
    max_bytes = _checked_limit(max_bytes)
    desired_b = _decode_map(desired, "desired", max_bytes)
    expected_b = _decode_map(expected, "expected", max_bytes)
    all_paths = set(desired_b) | set(expected_b)
    for path in all_paths:
        parts = path.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in all_paths:
                raise ValueError(f"file path is also an ancestor directory: {path}")
    rootfd = _open_root(root)
    try:
        current = _snapshot_fd(rootfd, max_bytes)
        if current != expected:
            raise RuntimeError("snapshot conflict: current workspace differs from expected")
        changes = sorted(set(desired_b) | set(expected_b))
        for rel in changes:
            if desired_b.get(rel) == expected_b.get(rel) and rel in desired_b:
                continue
            parent, name = _open_parent(rootfd, rel, create=rel in desired_b)
            try:
                actual = _file_bytes(parent, name, max_bytes)
                wanted = desired_b.get(rel)
                expected_data = expected_b.get(rel)
                if actual != expected_data:
                    raise RuntimeError(f"snapshot conflict at {rel}")
                if wanted is None:
                    os.unlink(name, dir_fd=parent)
                else:
                    temp_name = f".sync-{uuid4().hex}"
                    tempfd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                    try:
                        try:
                            offset = 0
                            while offset < len(wanted):
                                offset += os.write(tempfd, wanted[offset:])
                            os.fsync(tempfd)
                        finally:
                            os.close(tempfd)
                        if _file_bytes(parent, name, max_bytes) != expected_data:
                            raise RuntimeError(f"snapshot conflict at {rel}")
                        os.replace(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
                    finally:
                        try:
                            os.unlink(temp_name, dir_fd=parent)
                        except FileNotFoundError:
                            pass
            finally:
                os.close(parent)
    finally:
        os.close(rootfd)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("snapshot", "apply"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        payload = snapshot(args.root)
    else:
        raw = sys.stdin.buffer.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("JSON payload exceeds limit")
        obj = json.loads(raw)
        if not isinstance(obj, dict) or set(obj) != {"desired", "expected"}:
            raise ValueError("apply input must contain exactly desired and expected")
        apply_snapshot(args.root, obj["desired"], obj["expected"])
        payload = {}
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    _main()
