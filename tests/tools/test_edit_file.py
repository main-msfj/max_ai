"""edit_file replaces an exact text inside a file of the session's folders."""

import os
from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem_v2 import EditFileTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

CODE = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a + b\n"


@pytest.fixture
async def sandbox(tmp_path):
    """A user's sandbox; session ``s1`` has some files, session ``s2`` another."""
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    await bound.upload_file("s1", "calc.py", CODE.encode())
    await bound.upload_file("s1", "win.txt", b"one\r\ntwo\r\nthree\r\n")
    await bound.upload_file("s1", "latin.txt", "caf\xe9\n".encode("latin-1"))
    await bound.upload_file("s2", "secret.txt", b"another session")
    (tmp_path / "host.txt").write_text("outside the sandbox folders")
    return bound


@pytest.fixture
def workspace(sandbox):
    return Path(sandbox.session_paths("s1")["workspace"])


async def edit(sandbox, path, old, new, **more):
    parameters = {"path": path, "old_text": old, "new_text": new, **more}
    record = ToolCallRecord(tool_name="edit_file", parameters=parameters)
    ctx = RunContext(user_id="user-1", session_id="s1")
    return await ToolDispatcher(ToolRegistry([EditFileTool()]), sandbox).dispatch(record, ctx)


async def test_one_place_is_changed_and_the_rest_stays(sandbox, workspace):
    result = await edit(sandbox, "calc.py", "def sub(a, b):\n    return a + b", "def sub(a, b):\n    return a - b")
    assert result.success and result.result == "Edited calc.py: 1 place changed."
    assert (workspace / "calc.py").read_text() == CODE[:-6] + "a - b\n"
    assert not list(workspace.rglob("*.partial"))


async def test_a_text_that_repeats_needs_more_lines_or_replace_all(sandbox, workspace):
    result = await edit(sandbox, "calc.py", "a + b", "a * b")
    assert not result.success and "appears 2 times" in result.error
    assert (workspace / "calc.py").read_text() == CODE
    result = await edit(sandbox, "calc.py", "a + b", "a * b", replace_all=True)
    assert result.result == "Edited calc.py: 2 places changed."
    assert (workspace / "calc.py").read_text().count("a * b") == 2


async def test_an_empty_new_text_deletes(sandbox, workspace):
    assert (await edit(sandbox, "calc.py", "\ndef sub(a, b):\n    return a + b\n", "")).success
    assert (workspace / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


async def test_a_windows_file_keeps_its_line_breaks(sandbox, workspace):
    assert (await edit(sandbox, "win.txt", "one\ntwo", "1\n2")).success
    assert (workspace / "win.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"


async def test_an_edited_script_stays_executable(sandbox, workspace):
    os.chmod(workspace / "calc.py", 0o755)
    await edit(sandbox, "calc.py", "add", "plus")
    assert os.access(workspace / "calc.py", os.X_OK)


@pytest.mark.parametrize(
    "path, old, new, reason",
    [
        ("calc.py", "def mul", "x", "is not in the file"),
        ("calc.py", "add", "add", "are the same"),
        ("missing.py", "a", "b", "does not exist"),
        ("latin.txt", "caf", "tea", "not UTF-8 text"),
        ("../s2/secret.txt", "another", "x", "outside the folders"),
        ("../../../host.txt", "outside", "x", "outside the folders"),
    ],
)
async def test_what_cannot_be_edited_says_why(sandbox, workspace, tmp_path, path, old, new, reason):
    result = await edit(sandbox, path, old, new)
    assert not result.success and reason in result.error
    assert (workspace / "latin.txt").read_bytes() == b"caf\xe9\n"
    assert (tmp_path / "host.txt").read_text() == "outside the sandbox folders"


async def test_a_link_pointing_outside_is_refused(sandbox, workspace, tmp_path):
    os.symlink(tmp_path / "host.txt", workspace / "shortcut.txt")
    result = await edit(sandbox, "shortcut.txt", "outside", "inside")
    assert not result.success and "outside the folders" in result.error
    assert (tmp_path / "host.txt").read_text() == "outside the sandbox folders"
