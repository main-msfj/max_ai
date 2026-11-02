"""DockerExecutor copies the workspace in and out: no host path is shared."""

import hashlib
import os

import pytest

from max_ai.capabilities.executor.docker import DockerExecutor
from max_ai.capabilities.workspace.local import LocalWorkspace

pytestmark = pytest.mark.skipif(
    os.getenv("MAXAI_TEST_DOCKER") != "1", reason="needs Docker; set MAXAI_TEST_DOCKER=1",
)


async def test_files_go_in_come_back_and_deletions_follow(tmp_path):
    workspace = LocalWorkspace(root=tmp_path)
    directory = workspace.materialize("u1", "s1")
    (directory.workspace_dir / "notes.txt").write_text("from the host\n")
    executor = DockerExecutor()
    await executor.prepare()
    session = await executor.connect(workspace, "u1", "s1")
    try:
        await executor.sync(session, "to_environment")
        result = await executor.execute(
            session, "cd workspace && cat notes.txt && mkdir -p docs && echo x > docs/a.md"
            " && echo planted > ../jaja.md && id -u", timeout=60)
        assert result.exit_code == 0 and result.stdout.split() == ["from", "the", "host", "1000"]
        await executor.sync(session, "to_workspace")
        assert (directory.workspace_dir / "docs" / "a.md").read_text() == "x\n"
        assert not (directory.root / "jaja.md").exists()  # outside workspace/: stays inside

        (directory.workspace_dir / "notes.txt").unlink()
        await executor.sync(session, "to_environment")
        assert "notes.txt" not in (await executor.execute(session, "ls workspace", timeout=60)).stdout
    finally:
        await executor.clean(session)


async def test_binary_and_large_files_and_conflicts(tmp_path):
    workspace = LocalWorkspace(root=tmp_path)
    directory = workspace.materialize("u1", "s1")
    blob = os.urandom(12 << 20)  # past the old 8 MB limit, not text
    (directory.root / "workspace" / "big.bin").write_bytes(blob)
    (directory.root / "workspace" / "shared.txt").write_text("v1\n")
    executor = DockerExecutor()
    session = await executor.connect(workspace, "u1", "s1")
    try:
        await executor.sync(session, "to_environment")
        inside = await executor.execute(session, "sha256sum workspace/big.bin | cut -c1-64", timeout=60)
        assert inside.stdout.strip() == hashlib.sha256(blob).hexdigest()

        # The command deletes one file and writes a binary one: both come back.
        await executor.execute(session, "rm workspace/big.bin && head -c 3000 /dev/urandom > workspace/out.bin", timeout=60)
        await executor.sync(session, "to_workspace")
        assert not (directory.root / "workspace" / "big.bin").exists()
        assert (directory.root / "workspace" / "out.bin").stat().st_size == 3000

        # Changed on both sides since the last sync: the sync stops.
        await executor.execute(session, "echo sandbox > workspace/shared.txt", timeout=60)
        (directory.root / "workspace" / "shared.txt").write_text("host\n")
        with pytest.raises(RuntimeError, match="conflict: shared.txt"):
            await executor.sync(session, "to_environment")
    finally:
        await executor.clean(session)


async def test_a_killed_container_is_reported_lost_and_replaced(tmp_path):
    from max_ai.core.environment.manager import EnvironmentManager
    from max_ai.core.executor.process import run_process
    from max_ai.errors.executor import SandboxLost

    executor = DockerExecutor()
    await executor.prepare()
    async with EnvironmentManager(executor, LocalWorkspace(root=tmp_path)) as manager:
        with pytest.raises(SandboxLost):
            async with manager.acquire("u1", "s1") as session:
                await run_process(["docker", "kill", session.handle.name], timeout=30)
                await executor.execute(session, "echo hi", timeout=30)
        async with manager.acquire("u1", "s1") as session:
            assert (await executor.execute(session, "echo hi", timeout=30)).stdout == "hi\n"


async def test_bash_runs_its_script_in_the_container_and_stops_at_the_limit(tmp_path, monkeypatch):
    from max_ai.base.tools import ToolContext
    from max_ai.capabilities.tools.bash import BashTool
    from max_ai.config import setting
    from max_ai.types.tool_call import ToolCallRecord

    workspace = LocalWorkspace(root=tmp_path)
    directory = workspace.materialize("u1", "s1")
    executor = DockerExecutor()
    await executor.prepare()
    session = await executor.connect(workspace, "u1", "s1")
    tool = BashTool()

    async def bash(command):
        context = ToolContext("r1", session_id="s1", user_id="u1",
                              deps={"executor": executor, "execution_session": session})
        record = ToolCallRecord(tool_name="bash", parameters={"command": command, "description": "x"})
        return (await tool.execute(record, context)).result

    try:
        await executor.sync(session, "to_environment")
        made = await bash("mkdir -p docs && cd docs && echo x > a.md && pwd")
        assert made["output"] == "/workspaces/u1/workspace/docs"
        assert made["files"] == {"created": ["docs/a.md"]}
        monkeypatch.setattr(setting, "tool_timeout_seconds", 2)
        stopped = await bash("echo started; sleep 60")
        assert stopped["output"].startswith("started") and "Stopped after 2 seconds" in stopped["note"]
        await executor.sync(session, "to_workspace")
        assert (directory.workspace_dir / "docs" / "a.md").read_text() == "x\n"
    finally:
        await executor.clean(session)
