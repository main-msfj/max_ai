"""Explicit working copies of a user's workspace, independent of providers.

The owner must stop tools and serialize workspace writers before creating or
publishing a copy. This is file staging, not process isolation or a database
transaction. Publication replaces individual files atomically, not the whole
workspace. Symlinks and special files are rejected. File modes and removal of
empty directories are intentionally outside this content-only contract.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .workspace import WorkspaceBase


@dataclass(frozen=True)
class WorkspaceChanges:
    created: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return self.created + self.modified + self.deleted


class WorkspaceConflictError(RuntimeError):
    def __init__(self, paths: tuple[str, ...]):
        self.paths = paths
        super().__init__(f"Workspace changed concurrently: {', '.join(paths)}")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_tree(root: Path, max_bytes: int) -> tuple[dict[str, bytes], list[str]]:
    """Read bounded regular files without following symbolic links (POSIX)."""
    files: dict[str, bytes] = {}
    directories: list[str] = []
    remaining = max_bytes

    def walk(fd: int, prefix: str = "") -> None:
        nonlocal remaining
        for name in sorted(os.listdir(fd)):
            relative = f"{prefix}/{name}" if prefix else name
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    directories.append(relative)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(mode):
                file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(file_fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError(f"Not a regular file: {relative}")
                    content = stream.read(remaining + 1)
                remaining -= len(content)
                if remaining < 0:
                    raise ValueError("Workspace exceeds max_bytes")
                files[relative] = content
            else:
                raise ValueError(f"Symlink or special file rejected: {relative}")

    # Check every component, including ancestors of the supplied root.
    fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        walk(fd)
    finally:
        os.close(fd)
    return files, directories


class ExecutionWorkspace:
    """One user-scoped working copy retained until explicit publish/discard.

    ``workspace`` can be passed to an Executor in place of the persistent
    Workspace; it preserves the existing user/skills/conversation layout.
    No command failure, cancellation, destructor or context exit publishes it.
    The caller owns retention and cleanup. Do not expose this controller or its
    baseline to the model; mount only ``root`` into the execution environment.

    Requires exclusive access during snapshot/publication. The manager must
    coordinate other processes too if they share the persistent workspace.
    """

    def __init__(
        self,
        source: WorkspaceBase,
        user_id: str,
        conversation_id: str,
        *,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._source = source.materialize(user_id, conversation_id).root
        self._max_bytes = max_bytes
        files, directories = _read_tree(self._source, max_bytes)
        self._baseline = {path: _digest(data) for path, data in files.items()}
        self._container = Path(tempfile.mkdtemp(prefix="max-ai-workspace-")).resolve()
        self._closed = False
        try:
            if self._container.is_relative_to(source.base_root):
                raise ValueError("Temporary workspace must be outside the persistent workspace")
            self.workspace = type(source)(self._container)
            self.root = self.workspace.materialize(user_id, conversation_id).root
            for directory in directories:
                (self.root / directory).mkdir(parents=True, exist_ok=True)
            for path, data in files.items():
                (self.root / path).write_bytes(data)
        except BaseException:
            shutil.rmtree(self._container)
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Execution workspace has been discarded")

    def _changes(self, files: dict[str, bytes]) -> WorkspaceChanges:
        current = {path: _digest(data) for path, data in files.items()}
        return WorkspaceChanges(
            created=tuple(sorted(current.keys() - self._baseline.keys())),
            modified=tuple(sorted(
                path for path in current.keys() & self._baseline.keys()
                if current[path] != self._baseline[path]
            )),
            deleted=tuple(sorted(self._baseline.keys() - current.keys())),
        )

    def changes(self) -> WorkspaceChanges:
        """Inspect content changes, including binary files, without publishing."""
        self._check_open()
        files, _ = _read_tree(self.root, self._max_bytes)
        return self._changes(files)

    def publish(self) -> WorkspaceChanges:
        """Apply reviewed changes after conflict checking; never auto-approve.

        Unrelated concurrent edits are preserved. On failure the working copy
        remains available. An I/O failure can leave a partially applied batch;
        already applied content is accepted on retry. This method does not
        certify task correctness or replace an approval/verification gate.
        """
        self._check_open()
        desired, _ = _read_tree(self.root, self._max_bytes)
        changes = self._changes(desired)
        current, current_dirs = _read_tree(self._source, self._max_bytes)
        directories = set(current_dirs)
        conflicts = []
        for path in changes.paths:
            actual = _digest(current[path]) if path in current else None
            wanted = _digest(desired[path]) if path in desired else None
            if actual not in (self._baseline.get(path), wanted):
                conflicts.append(path)
            elif path in desired and (
                path in directories
                or any(str(parent) in current for parent in Path(path).parents if str(parent) != ".")
            ):
                # File/directory conversions need explicit resolution.
                conflicts.append(path)
        if conflicts:
            raise WorkspaceConflictError(tuple(sorted(conflicts)))

        for path in changes.created + changes.modified:
            data = desired[path]
            if current.get(path) == data:
                continue
            target = self._source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        for path in changes.deleted:
            if path in current:
                (self._source / path).unlink()
        self._baseline = {path: _digest(data) for path, data in desired.items()}
        return changes

    def discard(self) -> None:
        """Remove only this working copy, after its processes have stopped."""
        if not self._closed:
            shutil.rmtree(self._container)
            self._closed = True
