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
