"""Provider-independent session lifecycle for the Executor contract.

Local runs on the user's files directly. Docker and Modal hold a copy:
before each lease sync("to_environment") copies workspace/ and skills/ in,
after it sync("to_workspace") brings workspace/ back. The provider does the
transfer; this manager only decides when.

Connecting and syncing are retried while the provider can't be reached
(core/retry.py). A sandbox lost mid-lease (SandboxLost) is dropped without
syncing back, and the next acquire connects a new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from ...base.executor import ExecutionSession, ExecutorBase, SyncDirection
from ...base.workspace import WorkspaceBase
from ...config import setting
from ...errors.executor import SandboxLost
from ..retry import RetryPolicy, retrying

logger = logging.getLogger(__name__)


def _transient(error: BaseException) -> bool:
    """A dropped connection or a slow answer: worth trying again. A lost
    sandbox is not: it gets replaced instead."""
    return isinstance(error, (ConnectionError, TimeoutError)) and not isinstance(error, SandboxLost)


@dataclass
class _Entry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: ExecutionSession | None = None
    idle: asyncio.Task[None] | None = None


class EnvironmentManager:
    """One executor and workspace per manager, owned by one event loop.

    Leases serialize turns per user because conversations share editable files.
    Separate manager instances/processes require external coordination when
    writing to the same workspace.
    """

    def __init__(
        self,
        executor: ExecutorBase,
        workspace: WorkspaceBase,
        *,
        idle_timeout: float | None = None,
        retry: RetryPolicy | None = None,
    ):
        if idle_timeout is None:
            idle_timeout = setting.environment_idle_timeout
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError("idle_timeout must be finite and nonnegative")
        self.executor = executor
        self.workspace = workspace
        self.idle_timeout = idle_timeout
        self.retry = retry or RetryPolicy()
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._users: dict[str, asyncio.Lock] = {}
        self._closed = False
        self.cleanup_errors: dict[tuple[str, str], Exception] = {}

    def _validate(
        self, session: ExecutionSession, user_id: str, conversation_id: str
    ) -> None:
        if (session.user_id, session.conversation_id, session.workspace.base_root) != (
            user_id,
            conversation_id,
            self.workspace.base_root,
        ):
            raise ValueError(
                "Executor returned a session for a different workspace or identity"
            )

    @asynccontextmanager
    async def acquire(
        self, user_id: str, conversation_id: str
    ) -> AsyncIterator[ExecutionSession]:
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
                entry.session = await self._connect(user_id, conversation_id)
            try:
                # Recover a failed sync before handing the session back out.
                if key in self.cleanup_errors:
                    await self._sync(entry.session, "to_workspace")
                    self.cleanup_errors.pop(key, None)
                await self._sync(entry.session, "to_environment")
            except SandboxLost as lost:
                # It died while idle (e.g. Modal's lifetime): start a new one.
                logger.warning("Sandbox lost while idle, connecting a new one: %s", lost)
                await self._drop(key, entry)
                entry.session = await self._connect(user_id, conversation_id)
                await self._sync(entry.session, "to_environment")
            cancelled = lost = False
            try:
                yield entry.session
            except asyncio.CancelledError:
                cancelled = True
                raise
            except SandboxLost:
                lost = True
                raise
            finally:

                async def release():
                    try:
                        if lost:
                            await self._drop(key, entry)
                            return
                        if cancelled:
                            await self._disconnect(entry)
                        else:
                            await self._sync(entry.session, "to_workspace")
                    except SandboxLost:
                        await self._drop(key, entry)
                        raise
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

    async def _expire(self, key: tuple[str, str], entry: _Entry) -> None:
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

    async def _connect(self, user_id: str, conversation_id: str) -> ExecutionSession:
        """A new session, retried while the provider can't be reached."""

        async def attempt() -> ExecutionSession:
            session = await self.executor.connect(self.workspace, user_id, conversation_id)
            try:
                self._validate(session, user_id, conversation_id)
            except Exception:
                await self.executor.clean(session)
                raise
            return session

        return await retrying(attempt, policy=self.retry, transient=_transient, on_retry=_log_retry)

    async def _sync(self, session: ExecutionSession, direction: SyncDirection) -> None:
        """Sync, retried while the provider can't be reached: it compares
        hashes, so running it again is safe."""
        await retrying(
            lambda: self.executor.sync(session, direction),
            policy=self.retry, transient=_transient, on_retry=_log_retry,
        )

    async def _drop(self, key: tuple[str, str], entry: _Entry) -> None:
        """Forget a lost session: its runtime is gone, nothing to sync back."""
        session, entry.session = entry.session, None
        if key in self.cleanup_errors:
            logger.warning("Unsynced changes of a lost sandbox are gone: %s", key)
            self.cleanup_errors.pop(key, None)
        if session is not None:
            with contextlib.suppress(Exception):
                await self.executor.clean(session)

    async def _disconnect(self, entry: _Entry) -> None:
        if entry.session is not None:
            try:
                await self._sync(entry.session, "to_workspace")
            except SandboxLost:
                pass  # already gone: nothing to bring back
            else:
                await self.executor.disconnect(entry.session)
            with contextlib.suppress(SandboxLost):
                await self.executor.clean(entry.session)
            entry.session = None

    async def close(self) -> None:
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

    async def __aenter__(self) -> EnvironmentManager:
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


def _log_retry(attempt: int, error: BaseException, delay: float) -> None:
    logger.warning("Sandbox unreachable (try %d), retrying in %.1fs: %s", attempt, delay, error)
