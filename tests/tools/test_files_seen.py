"""The file tools remember what the model saw and start where bash ended."""

from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
async def sandbox(tmp_path):
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    await bound.upload_file("s1", "notes.md", b"first\n")
    await bound.upload_file("s1", "repo/src/app.py", b"print(1)\n")
    return bound


@pytest.fixture
def ctx():
    return RunContext(user_id="user-1", session_id="s1")


@pytest.fixture
def workspace(sandbox):
    return Path(sandbox.session_paths("s1")["workspace"])


async def call(sandbox, ctx, tool_name, **parameters):
    registry = ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()])
    record = ToolCallRecord(tool_name=tool_name, parameters=parameters)
    return await ToolDispatcher(registry, sandbox).dispatch(record, ctx)


async def test_reading_notes_the_file_under_its_whole_path(sandbox, ctx, workspace):
    await call(sandbox, ctx, "read_file", path="./repo/../notes.md", limit=1)
    assert list(ctx.files_seen) == [f"{workspace}/notes.md"]


async def test_a_file_changed_by_someone_else_must_be_read_again(sandbox, ctx, workspace):
    await call(sandbox, ctx, "read_file", path="notes.md")
    (workspace / "notes.md").write_text("changed by a command\n")  # bash, or the user
    refused = await call(sandbox, ctx, "write_file", path="notes.md", content="mine\n")
    assert not refused.success and "it changed since you read it; read it again" in refused.error
    assert (workspace / "notes.md").read_text() == "changed by a command\n"
    await call(sandbox, ctx, "read_file", path="notes.md")
    assert (await call(sandbox, ctx, "write_file", path="notes.md", content="mine\n")).success


async def test_what_the_model_wrote_or_edited_counts_as_seen(sandbox, ctx, workspace):
    assert (await call(sandbox, ctx, "write_file", path="new.txt", content="one\n")).success
    assert (await call(sandbox, ctx, "write_file", path="new.txt", content="two\n")).success
    assert (await call(sandbox, ctx, "edit_file", path="new.txt", old_text="two", new_text="three")).success
    assert (await call(sandbox, ctx, "write_file", path="new.txt", content="four\n")).success
    assert (workspace / "new.txt").read_text() == "four\n"


async def test_an_edit_does_not_need_a_read_but_notes_the_file(sandbox, ctx, workspace):
    assert (await call(sandbox, ctx, "edit_file", path="notes.md", old_text="first", new_text="second")).success
    assert (await call(sandbox, ctx, "write_file", path="notes.md", content="third\n")).success


async def test_relative_paths_start_where_bash_ended(sandbox, ctx, workspace):
    ctx.cwd = f"{workspace}/repo"
    assert (await call(sandbox, ctx, "read_file", path="src/app.py")).result.endswith("print(1)")
    assert (await call(sandbox, ctx, "write_file", path="src/new.py", content="x = 1\n")).success
    assert (await call(sandbox, ctx, "edit_file", path="src/app.py", old_text="1", new_text="2")).success
    assert (workspace / "repo/src/new.py").exists()
    assert (workspace / "repo/src/app.py").read_text() == "print(2)\n"
    assert f"{workspace}/repo/src/app.py" in ctx.files_seen
