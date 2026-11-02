"""The EnvironmentManager retries an unreachable provider and replaces a lost sandbox."""

from __future__ import annotations

import pytest

from max_ai.base.executor import ExecutionResult, ExecutionSession, ExecutorBase
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.environment.manager import EnvironmentManager
from max_ai.core.retry import RetryPolicy
from max_ai.errors.executor import SandboxLost


class FlakyExecutor(ExecutorBase):
    """Sessions s1, s2, ...; ``dead`` ones raise SandboxLost like Docker or Modal."""

    def __init__(self, unreachable: int = 0):
        self.unreachable = unreachable
        self.connects = 0
        self.dead: set[str] = set()
        self.synced: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    async def connect(self, workspace, user_id, conversation_id) -> ExecutionSession:
        self.connects += 1
        if self.unreachable:
            self.unreachable -= 1
            raise ConnectionError("provider unreachable")
        return ExecutionSession(f"s{self.connects}", user_id, conversation_id, workspace, "/w")

    async def sync(self, session, direction) -> None:
        if session.id in self.dead:
            raise SandboxLost("container is not running")
        self.synced.append((session.id, direction))

    async def execute(self, session, command, *, timeout=60, cancellation_token=None):
        if session.id in self.dead:
            raise SandboxLost("container is not running")
        return ExecutionResult("ok", "", 0)

    async def disconnect(self, session) -> None:
        pass

    async def clean(self, session) -> None:
        self.cleaned.append(session.id)


def _manager(executor: FlakyExecutor, tmp_path) -> EnvironmentManager:
    return EnvironmentManager(executor, LocalWorkspace(root=tmp_path),
                              retry=RetryPolicy(base_delay=0), idle_timeout=60)


async def test_an_unreachable_provider_is_retried(tmp_path):
    executor = FlakyExecutor(unreachable=2)
    async with _manager(executor, tmp_path) as manager:
        async with manager.acquire("u", "c") as session:
            assert session.id == "s3"


async def test_a_sandbox_lost_mid_command_is_replaced_for_the_next_one(tmp_path):
    executor = FlakyExecutor()
    async with _manager(executor, tmp_path) as manager:
        with pytest.raises(SandboxLost, match="Workspace files are kept"):
            async with manager.acquire("u", "c") as session:
                executor.dead.add(session.id)
                await executor.execute(session, "pip install x")
        assert executor.cleaned == ["s1"]
        assert ("s1", "to_workspace") not in executor.synced  # nothing to bring back
        async with manager.acquire("u", "c") as session:
            assert session.id == "s2"
            assert (await executor.execute(session, "ls")).stdout == "ok"


async def test_a_sandbox_that_died_while_idle_is_replaced(tmp_path):
    executor = FlakyExecutor()
    async with _manager(executor, tmp_path) as manager:
        async with manager.acquire("u", "c"):
            pass
        executor.dead.add("s1")  # e.g. Modal's lifetime ran out
        async with manager.acquire("u", "c") as session:
            assert session.id == "s2"
        assert ("s2", "to_environment") in executor.synced
