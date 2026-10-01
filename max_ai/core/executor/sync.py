"""Keep the user's folder on the host and its copy in a sandbox the same.

Docker and Modal share it; LocalExecutor works on the host folder itself.
Only ``workspace/`` comes back from the sandbox. ``skills/`` goes one way:
whatever the model changes there is overwritten before the next command and
never reaches the host. Harness files (``.maxai-*``) never leave the host.
Both sides list their regular files with a SHA-256 each, and only what
changed moves, as a tar in base64. Inside the sandbox it is plain bash
(``find``, ``sha256sum``, ``tar``, ``base64``): nothing of max_ai.

The host remembers the hashes of the last sync (the baseline). A file that
changed on both sides since then is a conflict: the sync stops instead of
overwriting either side.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tarfile
import typing as t
from pathlib import Path, PurePosixPath

from ...base.executor import ExecutionResult, SyncDirection
from ...types.workspace import WorkspaceDirectory

MAX_SYNC_BYTES = 100 * 1024 * 1024  # files moved in one sync, before base64

# Every script gets the sandbox copy's root as $1.
_HASHES = (
    'cd "$1" 2>/dev/null || exit 0;'
    " find . -type f ! -name '.maxai-*' -print0 | xargs -0 -r sha256sum -z"
)
_PACK = 'cd "$1" && tar -c --null -T - | base64 -w0'  # paths on stdin, NUL-separated
_UNPACK = (  # the tar on stdin; paths to delete as $2...
    'mkdir -p "$1" && cd "$1" && base64 -d | tar -x --no-same-owner'
    ' && shift && { [ $# -eq 0 ] || rm -f -- "$@"; }'
)


class Run(t.Protocol):
    """Runs ``bash -c script maxai root *args`` in the sandbox."""

    async def __call__(
        self, script: str, args: list[str], *, stdin: str | None = None, limit: int = 1 << 20
    ) -> ExecutionResult: ...


async def sync_session(
    run: Run, directory: WorkspaceDirectory, sandbox_root: str, baseline: dict[str, str],
    direction: SyncDirection,
) -> None:
    """Sync one session's folder: ``directory`` is its ``WorkspaceDirectory``
    on the host, ``sandbox_root`` the same folder in the sandbox."""
    sandbox = PurePosixPath(sandbox_root)
    await sync_workspace(
        run, directory.workspace_dir, str(sandbox / "workspace"), baseline, direction
    )
    if direction == "to_environment":
        await mirror(run, directory.skill_dir, str(sandbox / "skills"))


async def mirror(run: Run, host_root: Path, sandbox_root: str) -> None:
    """Make the sandbox copy of ``host_root`` exactly the host's, one way."""
    host = host_hashes(host_root)
    sandbox = _parse_hashes(await _checked(run(_HASHES, [sandbox_root], limit=16 << 20), "list"))
    copy = [name for name in sorted(host) if sandbox.get(name) != host[name]]
    delete = [name for name in sorted(sandbox) if name not in host]
    if copy or delete:
        await _checked(
            run(_UNPACK, [sandbox_root, *delete], stdin=pack(host_root, copy)), "upload"
        )


async def sync_workspace(
    run: Run, host_root: Path, sandbox_root: str, baseline: dict[str, str], direction: SyncDirection
) -> None:
    """Copy what changed since ``baseline`` in the ``direction`` given, and
    move ``baseline`` forward for the files that are now the same."""
    if direction not in ("to_environment", "to_workspace"):
        raise ValueError("Unknown synchronization direction")
    host = host_hashes(host_root)
    sandbox = _parse_hashes(await _checked(run(_HASHES, [sandbox_root], limit=16 << 20), "list"))
    source, destination = (host, sandbox) if direction == "to_environment" else (sandbox, host)
    copy, delete = plan(source, destination, baseline)

    if copy or delete:
        if direction == "to_environment":
            await _checked(
                run(_UNPACK, [sandbox_root, *delete], stdin=pack(host_root, copy)), "upload"
            )
        else:
            if copy:
                paths = "\0".join(copy) + "\0"
                limit = MAX_SYNC_BYTES * 4 // 3 + 4096
                unpack(host_root, await _checked(run(_PACK, [sandbox_root], stdin=paths, limit=limit), "download"))
            remove(host_root, delete)

    # Only files now equal on both sides move forward; a change on the
    # other side that this pass didn't carry still differs next time.
    arrived = {**destination, **{name: source[name] for name in copy}}
    for name in delete:
        arrived.pop(name, None)
    for name in set(source) | set(arrived) | set(baseline):
        if source.get(name) == arrived.get(name):
            if name in source:
                baseline[name] = source[name]
            else:
                baseline.pop(name, None)


def plan(
    source: dict[str, str], destination: dict[str, str], baseline: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Files to copy from ``source`` and to delete in ``destination``.

    A file is carried over when ``source`` changed it since ``baseline``;
    if ``destination`` changed it too, to something else, that's a conflict.
    """
    copy: list[str] = []
    delete: list[str] = []
    for name in sorted(set(source) | set(baseline)):
        before, incoming, current = baseline.get(name), source.get(name), destination.get(name)
        if incoming == before or incoming == current:
            continue
        if current != before:
            raise RuntimeError(f"Workspace sync conflict: {name}")
        if incoming is None:
            delete.append(name)
        else:
            copy.append(name)
    return copy, delete


# -------- HOST SIDE ------------------------------------------------------------
def host_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file below ``root``; symlinks and harness
    files (``.maxai-*``) are skipped."""
    hashes: dict[str, str] = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(folder, d))]
        for name in files:
            if name.startswith(".maxai-"):
                continue
            path = os.path.join(folder, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as file:
                for chunk in iter(lambda: file.read(1 << 20), b""):
                    digest.update(chunk)
            hashes[Path(path).relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def pack(root: Path, paths: list[str]) -> str:
    """A tar of ``paths`` below ``root``, in base64."""
    buffer = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in paths:
            path = _inside(root, name)
            total += path.stat().st_size
            if total > MAX_SYNC_BYTES:
                raise RuntimeError(f"Workspace sync moves more than {MAX_SYNC_BYTES >> 20} MB")
            tar.add(path, arcname=name, recursive=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def unpack(root: Path, data: str) -> None:
    """Extract a base64 tar made in the sandbox: regular files only, all of
    them inside ``root`` (the ``data`` filter refuses ``..``, absolute paths
    and links out)."""
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(data)), mode="r:") as tar:
        members = tar.getmembers()
        for member in members:
            _inside(root, member.name)
            if not member.isfile():
                raise RuntimeError(f"Workspace sync refused {member.name}: not a regular file")
        tar.extractall(root, members=members, filter="data")


def remove(root: Path, paths: list[str]) -> None:
    """Delete ``paths`` below ``root``; already gone is fine."""
    for name in paths:
        _inside(root, name).unlink(missing_ok=True)


def _inside(root: Path, name: str) -> Path:
    """``root / name``, refusing any path that would end up elsewhere."""
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise RuntimeError(f"Workspace sync refused path {name!r}")
    path = root.joinpath(*parts)
    if not path.parent.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise RuntimeError(f"Workspace sync refused path {name!r}")
    return path


# -------- SANDBOX SIDE -----------------------------------------------------------
def _parse_hashes(output: str) -> dict[str, str]:
    """``sha256sum -z`` output: ``<hash>  ./<path>`` records, NUL-ended."""
    hashes: dict[str, str] = {}
    for record in output.split("\0"):
        if record:
            digest, _, path = record.partition("  ")
            hashes[path.removeprefix("./")] = digest
    return hashes


async def _checked(pending: t.Awaitable[ExecutionResult], step: str) -> str:
    """The command's stdout, or an error saying which sync step failed."""
    result = await pending
    if result.exit_code != 0 or result.timed_out or result.truncated:
        raise RuntimeError(f"Workspace sync ({step}) failed: {result.stderr[-2000:]}")
    return result.stdout
