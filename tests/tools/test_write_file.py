"""write_file creates or replaces a whole text file in the session's folders."""

import os
from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import ReadFileTool, WriteFileTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

CTX: list[RunContext] = []  # the conversation of the test in course


@pytest.fixture(autouse=True)
def ctx():
    CTX[:] = [RunContext(user_id="user-1", session_id="s1")]
    return CTX[0]


@pytest.fixture
async def sandbox(tmp_path):
    """A user's sandbox; session ``s1`` has one file, session ``s2`` another."""
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    await bound.upload_file("s1", "app.py", b"print('old')\n")
    await bound.upload_file("s2", "secret.txt", b"another session")
    (tmp_path / "host.txt").write_text("outside the sandbox folders")
    return bound


@pytest.fixture
def workspace(sandbox):
    return Path(sandbox.session_paths("s1")["workspace"])


async def write(sandbox, path, content):
    record = ToolCallRecord(tool_name="write_file", parameters={"path": path, "content": content})
    ctx = CTX[0]
    return await ToolDispatcher(ToolRegistry([WriteFileTool()]), sandbox).dispatch(record, ctx)


async def test_a_new_file_is_created_with_its_folders(sandbox, workspace):
    result = await write(sandbox, "src/utils/math.js", "export const two = 2\n// ñ, 日本\n")
    assert result.success and result.result == "Created src/utils/math.js (2 lines)."
    assert (workspace / "src/utils/math.js").read_text("utf-8") == "export const two = 2\n// ñ, 日本\n"
    assert not list(workspace.rglob("*.partial"))


async def read(sandbox, path):
    record = ToolCallRecord(tool_name="read_file", parameters={"path": path})
    return await ToolDispatcher(ToolRegistry([ReadFileTool()]), sandbox).dispatch(record, CTX[0])


async def test_an_existing_file_is_replaced_whole_once_it_was_read(sandbox, workspace):
    refused = await write(sandbox, "app.py", "print('new')\n")
    assert not refused.success and "you have not read it; read it first" in refused.error
    assert (workspace / "app.py").read_text() == "print('old')\n"
    await read(sandbox, "app.py")
    result = await write(sandbox, "app.py", "print('new')\n")
    assert result.result == "Replaced app.py (1 lines)."
    assert (workspace / "app.py").read_text() == "print('new')\n"


async def test_a_replaced_script_stays_executable(sandbox, workspace):
    os.chmod(workspace / "app.py", 0o755)
    await read(sandbox, "app.py")
    assert (await write(sandbox, "app.py", "#!/usr/bin/env python3\n")).success
    assert os.access(workspace / "app.py", os.X_OK)


async def test_a_big_file_arrives_complete(sandbox, workspace):
    content = "0123456789\n" * 50_000  # over the size an argument can have
    assert (await write(sandbox, "big.csv", content)).success
    assert (workspace / "big.csv").read_text() == content


async def test_the_tmp_folder_of_the_session_can_be_written(sandbox):
    tmp = sandbox.session_paths("s1")["tmp"]
    assert (await write(sandbox, f"{tmp}/notes.md", "# notes")).success
    assert Path(tmp, "notes.md").read_text() == "# notes"


@pytest.mark.parametrize(
    "path, reason",
    [
        ("../s2/secret.txt", "outside the folders"),
        ("../../../host.txt", "outside the folders"),
        ("/etc/evil.txt", "outside the folders"),
        ("skill", "outside the folders"),
        (".", "outside the folders"),
    ],
)
async def test_what_cannot_be_written_says_why(sandbox, tmp_path, path, reason):
    if path == "skill":
        path = f"{sandbox.layout.skills}/mine/SKILL.md"
    result = await write(sandbox, path, "x")
    assert not result.success and reason in result.error
    assert (tmp_path / "host.txt").read_text() == "outside the sandbox folders"


async def test_a_folder_is_not_replaced(sandbox, workspace):
    (workspace / "docs").mkdir()
    result = await write(sandbox, "docs", "x")
    assert not result.success and "is a folder" in result.error


async def test_a_link_pointing_outside_is_refused(sandbox, workspace, tmp_path):
    os.symlink(tmp_path / "host.txt", workspace / "shortcut.txt")
    os.symlink(tmp_path, workspace / "out")
    for path in ("shortcut.txt", "out/new.txt"):
        result = await write(sandbox, path, "changed")
        assert not result.success and "outside the folders" in result.error
    assert (tmp_path / "host.txt").read_text() == "outside the sandbox folders"
    assert not (tmp_path / "new.txt").exists()
