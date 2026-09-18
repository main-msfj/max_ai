"""Provider-independent session lifecycle for the Executor contract.

No working copy: the executor connects directly to the persistent workspace.
Local and Docker already share the filesystem (bind mount); a genuinely
remote provider stages its own transfer inside connect()/sync(), not here.
Safety comes from git discipline (status/diff before destructive ops), the
same discipline this harness expects of the model itself.
"""

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from ...base.executor import ExecutorBase, ExecutionSession
from ...base.workspace import WorkspaceBase


@dataclass
class _Entry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: ExecutionSession | None = None
    idle: asyncio.Task | None = None


class EnvironmentManager:
    """One executor and workspace per manager, owned by one event loop.

    Leases serialize turns per user because conversations share editable files.
    Separate manager instances/processes require external coordination when
    writing to the same workspace.
    """

    def __init__(self, executor: ExecutorBase, workspace: WorkspaceBase, *, idle_timeout: float = 300):
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError("idle_timeout must be finite and nonnegative")
        self.executor = executor
        self.workspace = workspace
        self.idle_timeout = idle_timeout
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._users: dict[str, asyncio.Lock] = {}
        self._closed = False
        self.cleanup_errors: dict[tuple[str, str], Exception] = {}

    def _validate(self, session, user_id, conversation_id):
        if (session.user_id, session.conversation_id, session.workspace.base_root) != (
            user_id, conversation_id, self.workspace.base_root,
        ):
            raise ValueError("Executor returned a session for a different workspace or identity")

    @asynccontextmanager
    async def acquire(self, user_id: str, conversation_id: str):
        """Connect lazily to the persistent workspace; reuse an open session.

        Callers should acquire only when execution is needed, not for text-only
        turns. The caller executes through executor.execute(session, ...).
        """
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        self.workspace.materialize(user_id, conversation_id)
        key = (user_id, conversation_id)
        entry = self._entries.setdefault(key, _Entry())
        user_lock = self._users.setdefault(user_id, asyncio.Lock())
        async with user_lock, entry.lock:
            if self._closed:
                raise RuntimeError("EnvironmentManager is closed")
            if entry.idle is not None:
                entry.idle.cancel()
                await asyncio.gather(entry.idle, return_exceptions=True)
                entry.idle = None
            if entry.session is None:
                session = await self.executor.connect(self.workspace, user_id, conversation_id)
                try:
                    self._validate(session, user_id, conversation_id)
                except Exception:
                    await self.executor.clean(session)
                    raise
                entry.session = session
            # Recover a failed sync before handing the session back out.
            if key in self.cleanup_errors:
                await self.executor.sync(entry.session, "to_workspace")
                self.cleanup_errors.pop(key, None)
            await self.executor.sync(entry.session, "to_environment")
            cancelled = False
            try:
                yield entry.session
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                async def release():
                    try:
                        if cancelled:
                            await self._disconnect(entry)
                        else:
                            await self.executor.sync(entry.session, "to_workspace")
                    except Exception as error:
                        self.cleanup_errors[key] = error
                        raise
                    else:
                        self.cleanup_errors.pop(key, None)
                        if entry.session is not None:
                            entry.idle = asyncio.create_task(self._expire(key, entry))

                cleanup = asyncio.create_task(release())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    async def _expire(self, key, entry):
        await asyncio.sleep(self.idle_timeout)
        async with self._users[key[0]], entry.lock:
            try:
                cleanup = asyncio.create_task(self._disconnect(entry))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
                self.cleanup_errors.pop(key, None)
            except Exception as error:
                self.cleanup_errors[key] = error

    async def _disconnect(self, entry):
        if entry.session is not None:
            await self.executor.sync(entry.session, "to_workspace")
            await self.executor.disconnect(entry.session)
            await self.executor.clean(entry.session)
            entry.session = None

    async def rebuild(self, user_id: str, conversation_id: str):
        """Recreate an inactive session after saving its workspace changes."""
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        key = (user_id, conversation_id)
        entry = self._entries[key]
        async with self._users[user_id], entry.lock:
            if self._closed:
                raise RuntimeError("EnvironmentManager is closed")
            if entry.session is None:
                raise RuntimeError("No connected session to rebuild")
            await self.executor.sync(entry.session, "to_workspace")
            previous = entry.session
            try:
                replacement = await self.executor.rebuild(previous)
            except BaseException:
                # The workspace was synced above. A failed recreation must
                # not leave the manager reusing the now-closed old session.
                await self.executor.clean(previous)
                entry.session = None
                raise
            try:
                self._validate(replacement, user_id, conversation_id)
            except Exception:
                await self.executor.clean(replacement)
                entry.session = None
                raise
            entry.session = replacement
            await self.executor.sync(replacement, "to_environment")
            self.cleanup_errors.pop(key, None)

    async def close(self):
        """Wait for leases and release every runtime, preserving workspace files.

        Failed entries remain available for another close attempt.
        """
        self._closed = True
        for entry in self._entries.values():
            if entry.idle is not None:
                entry.idle.cancel()
        await asyncio.gather(
            *(entry.idle for entry in self._entries.values() if entry.idle is not None),
            return_exceptions=True,
        )
        errors = []
        for key, entry in self._entries.items():
            async with self._users[key[0]], entry.lock:
                # An active lease may have scheduled its timer while close
                # was waiting for its lock.
                if entry.idle is not None and not entry.idle.done():
                    entry.idle.cancel()
                    await asyncio.gather(entry.idle, return_exceptions=True)
                try:
                    await self._disconnect(entry)
                    self.cleanup_errors.pop(key, None)
                except Exception as error:
                    self.cleanup_errors[key] = error
                    errors.append(error)
        if errors:
            raise ExceptionGroup("Environment cleanup failed", errors)

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        return self

    async def __aexit__(self, *args):
        await self.close()
