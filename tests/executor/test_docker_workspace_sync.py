"""DockerExecutor copies the workspace in and out: no host path is shared."""

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
    (directory.root / "notes.txt").write_text("from the host\n")
    executor = DockerExecutor()
    await executor.prepare()
    session = await executor.connect(workspace, "u1", "s1")
    try:
        await executor.sync(session, "to_environment")
        result = await executor.execute(session, "cat notes.txt && mkdir -p docs && echo x > docs/a.md && id -u")
        assert result.exit_code == 0 and result.stdout.split() == ["from", "the", "host", "1000"]
        await executor.sync(session, "to_workspace")
        assert (directory.root / "docs" / "a.md").read_text() == "x\n"

        (directory.root / "notes.txt").unlink()
        await executor.sync(session, "to_environment")
        assert "notes.txt" not in (await executor.execute(session, "ls")).stdout
    finally:
        await executor.clean(session)


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
