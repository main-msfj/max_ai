"""Application-scoped ownership of reusable conversation environments."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .environment import Environment
from ...base.workspace import Workspace

EnvironmentFactory = Callable[[Workspace, str, str], Environment]


@dataclass
class _Entry:
    factory: EnvironmentFactory  # Retain identity for the lifetime of its cache key.
    environment: Environment
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EnvironmentManager:
    """One application/event loop. Factories must keep their configuration stable.

    Different factory objects intentionally never share a container. Sharing a
    manager and factory allows multiple agent instances to reuse a conversation.
    Idle container removal belongs to the backend; lightweight entries remain
    cached until application shutdown.
    """

    def __init__(self):
        self._entries: dict[tuple[str, int, str, str], _Entry] = {}
        self._closed = False

    @asynccontextmanager
    async def acquire(self, factory: EnvironmentFactory, workspace: Workspace,
                      user_id: str, conversation_id: str):
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        key = (str(workspace.base_root.resolve()), id(factory), user_id, conversation_id)
        entry = self._entries.get(key)
        if entry is None:
            environment = factory(workspace, user_id, conversation_id)
            if not isinstance(environment, Environment):
                raise TypeError("Environment factory must return Environment")
            if (environment.user_id, environment.conversation_id,
                environment.workspace.base_root.resolve()) != (
                    user_id, conversation_id, workspace.base_root.resolve()):
                raise ValueError("Environment factory returned a different workspace or identity")
            entry = self._entries[key] = _Entry(factory, environment)
        async with entry.lock:
            if self._closed:
                raise RuntimeError("EnvironmentManager is closed")
            try:
                yield entry.environment  # Lazy: no container startup for text-only turns.
            finally:
                cleanup = asyncio.create_task(entry.environment.release())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    async def close(self) -> None:
        """Reject new leases, wait for active turns, and stop every owned backend."""
        self._closed = True

        async def stop(entry):
            async with entry.lock:
                await entry.environment.stop()

        results = await asyncio.gather(
            *(stop(entry) for entry in self._entries.values()), return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("Environment cleanup failed", errors)
        self._entries.clear()

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("EnvironmentManager is closed")
        return self

    async def __aexit__(self, *args):
        await self.close()
