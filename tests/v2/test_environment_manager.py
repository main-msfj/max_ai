import asyncio
from dataclasses import dataclass

import pytest
import pytest_asyncio

from max_ai.base.environment import ExecutionResult
from max_ai.base.executor import ExecutionSession
from max_ai.core.environment.manager import EnvironmentManager


@pytest_asyncio.fixture
async def manager_factory(workspace):
    managers = []
    def create(executor, **kwargs):
        manager = EnvironmentManager(executor, workspace, **kwargs)
        managers.append((manager, executor))
        return manager
    try:
        yield create
    finally:
        for manager, executor in managers:
            if hasattr(executor, "fail_download"):
                executor.fail_download = False
            try:
                await manager.close()
            finally:
                for user_id, conversation_id in list(manager._entries):
                    await manager.discard(user_id, conversation_id)


@dataclass
class FakeExecutor:
    connects: int = 0
    disconnects: int = 0
    cleans: int = 0
    syncs: list = None

    def __post_init__(self):
        self.syncs = [] if self.syncs is None else self.syncs

    async def connect(self, workspace, user_id, conversation_id):
        self.connects += 1
        directory = workspace.materialize(user_id, conversation_id)
        assert directory.root.is_dir()
        assert directory.root == workspace.base_root / user_id
        assert directory.conversation_dir == directory.root / conversation_id
        session = ExecutionSession(str(self.connects), user_id, conversation_id, workspace,
                                   str(directory.root), directory)
        self.session = session
        return session

    async def sync(self, session, direction):
        self.syncs.append(direction)

    async def disconnect(self, session):
        self.disconnects += 1

    async def clean(self, session):
        self.cleans += 1


@pytest.mark.asyncio
async def test_manager_reuses_session_and_retains_working_copy(workspace, manager_factory):
    executor = FakeExecutor()
    manager = manager_factory(executor, idle_timeout=60)
    async with manager.acquire("u", "c"):
        pass
    manager._entries[("u", "c")].working_copy.root.joinpath("survives").write_text("yes")
    async with manager.acquire("u", "c"):
        pass
    assert executor.connects == 1
    assert executor.syncs.count("to_environment") == 2
    assert manager._entries[("u", "c")].working_copy.root.joinpath("survives").read_text() == "yes"


@pytest.mark.asyncio
async def test_manager_publish_is_explicit_and_close_does_not_publish(workspace, manager_factory):
    executor = FakeExecutor()
    manager = manager_factory(executor, idle_timeout=60)
    async with manager.acquire("u", "c"):
        manager._entries[("u", "c")].working_copy.root.joinpath("x").write_text("x")
    await manager.close()
    assert not workspace.materialize("u", "c").root.joinpath("x").exists()
    await manager.publish("u", "c")
    assert workspace.materialize("u", "c").root.joinpath("x").read_text() == "x"


@pytest.mark.asyncio
async def test_cancellation_disconnects_without_publishing(workspace, manager_factory):
    executor = FakeExecutor()
    manager = manager_factory(executor, idle_timeout=60)
    with pytest.raises(asyncio.CancelledError):
        async with manager.acquire("u", "c"):
            manager._entries[("u", "c")].working_copy.root.joinpath("x").write_text("x")
            raise asyncio.CancelledError
    assert executor.disconnects == 1
    assert not workspace.materialize("u", "c").root.joinpath("x").exists()


@pytest.mark.asyncio
async def test_idle_expiry_disconnects_and_reconnects(workspace, manager_factory):
    executor = FakeExecutor()
    manager = manager_factory(executor, idle_timeout=0.001)
    async with manager.acquire("u", "c"):
        pass
    idle = manager._entries[("u", "c")].idle
    await asyncio.wait_for(asyncio.shield(idle), timeout=1)
    assert executor.disconnects == 1
    async with manager.acquire("u", "c"):
        pass
    assert executor.connects == 2


@pytest.mark.asyncio
async def test_sync_failure_records_error_and_preserves_copy(workspace, manager_factory):
    class Failing(FakeExecutor):
        fail_download = True
        async def sync(self, session, direction):
            if direction == "to_workspace" and self.fail_download:
                raise RuntimeError("download failed")
            return await super().sync(session, direction)
    executor = Failing()
    manager = manager_factory(executor)
    with pytest.raises(RuntimeError):
        async with manager.acquire("u", "c"):
            pass
    assert ("u", "c") in manager.cleanup_errors
    assert manager._entries[("u", "c")].working_copy is not None
    executor.fail_download = False
