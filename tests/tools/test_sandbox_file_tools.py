"""read_file, write_file and edit_file run one script in the sandbox and
never leave its workspace."""

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import filesystem_tools
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
def tools(tmp_path):
    read, write, edit = filesystem_tools(LocalSandbox(root=str(tmp_path)))
    return {"read": read, "write": write, "edit": edit}


async def call(tool, **parameters):
    return await tool.execute(ToolCallRecord(id="c", tool_name=tool.name, parameters=parameters))


async def test_write_then_read_inside_the_workspace(tools, tmp_path):
    written = await call(tools["write"], path="notes/a.txt", content="one\ntwo\nthree\n")
    assert written.success
    assert (tmp_path / ".agent/workspace/notes/a.txt").read_text() == "one\ntwo\nthree\n"
    part = await call(tools["read"], path="notes/a.txt", offset=1, limit=1)
    assert part.result == {"content": "two\n", "total_lines": 3}


async def test_edit_replaces_one_exact_piece(tools):
    await call(tools["write"], path="a.txt", content="hola mundo")
    assert (await call(tools["edit"], path="a.txt", old_string="mundo", new_string="max")).success
    assert (await call(tools["read"], path="a.txt")).result["content"] == "hola max"


async def test_edit_refuses_a_missing_or_repeated_piece(tools):
    await call(tools["write"], path="a.txt", content="x x")
    missing = await call(tools["edit"], path="a.txt", old_string="y", new_string="z")
    repeated = await call(tools["edit"], path="a.txt", old_string="x", new_string="z")
    assert "not found" in missing.error and "appears 2 times" in repeated.error
    everyone = await call(tools["edit"], path="a.txt", old_string="x", new_string="z", replace_all=True)
    assert everyone.result == {"replaced": 2}


async def test_a_path_outside_the_workspace_is_refused(tools, tmp_path):
    result = await call(tools["write"], path="../escape.txt", content="no")
    assert not result.success and "outside the workspace" in result.error
    assert not (tmp_path / ".agent/escape.txt").exists()


async def test_local_sandbox_stops_a_command_at_its_timeout(tmp_path):
    result = await LocalSandbox(root=str(tmp_path)).execute_command("sleep 5", timeout=0.2)
    assert result.timed_out and result.exit_code is None
