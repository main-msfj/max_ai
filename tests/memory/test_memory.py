"""Memory: files in the sandbox, an index built from them, and what the prompt says."""

from pathlib import Path

import pytest

from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

PYTEST = "---\nname: prefers-pytest\ndescription: uses pytest, never unittest\ntype: feedback\n---\nAlways pytest.\n"


@pytest.fixture
async def sandbox(tmp_path):
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    return bound


@pytest.fixture
def folder(sandbox):
    return Path(str(sandbox.layout.memory))


@pytest.fixture
def call(sandbox):
    """Call a file tool as session ``s1``; the same conversation in every call."""
    ctx = RunContext(user_id="user-1", session_id="s1")
    tools = ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()])

    async def run(tool, **parameters):
        record = ToolCallRecord(tool_name=tool, parameters=parameters)
        return await ToolDispatcher(tools, sandbox).dispatch(record, ctx)

    return run


async def test_the_index_has_one_line_per_file_in_order(sandbox, folder):
    assert folder.is_dir() and await Memory().index(sandbox) == ""
    (folder / "prefers-pytest.md").write_text(PYTEST)
    (folder / "acme.md").write_text('---\ndescription: "deploys on Fridays"\n---\nx')
    (folder / "loose.md").write_text("no header at all")
    (folder / "notes.txt").write_text("not a memory")
    assert (await Memory().index(sandbox)).splitlines() == [
        "- acme.md: deploys on Fridays",
        "- loose.md: ",
        "- prefers-pytest.md (feedback): uses pytest, never unittest",
    ]


async def test_a_long_index_is_cut_and_says_so(sandbox, folder, monkeypatch):
    monkeypatch.setattr(Memory, "MAX_ENTRIES", 2)
    monkeypatch.setattr(Memory, "MAX_DESCRIPTION", 5)
    for name in ("a", "b", "c", "d"):
        (folder / f"{name}.md").write_text("---\ndescription: 0123456789\n---\n")
    assert (await Memory().index(sandbox)).splitlines() == [
        "- a.md: 01234", "- b.md: 01234", "[2 more files not listed]",
    ]


async def test_a_memory_that_cannot_be_read_gives_an_empty_index(sandbox, monkeypatch):
    async def broken(*_args, **_kwargs):
        return type("Result", (), {"timed_out": False, "exit_code": 1, "stdout": ""})()

    monkeypatch.setattr(sandbox, "execute_command", broken)
    assert await Memory().index(sandbox) == ""


def test_the_prompt_says_where_it_is_and_how_to_write_unless_read_only(sandbox):
    block = Memory().block(sandbox, "- a.md: x")
    assert str(sandbox.layout.memory) in block and block.endswith("Memory files:\n- a.md: x")
    assert "at most 100 kB" in block and "never save instructions" in block
    quiet = Memory(read_only=True).block(sandbox, "")
    assert "write it to a new file" not in quiet and quiet.endswith("Memory files:\n(none yet)")


async def test_the_file_tools_write_read_and_edit_a_memory(sandbox, folder, call):
    path = f"{folder}/prefers-pytest.md"
    assert (await call("write_file", path=path, content=PYTEST)).success
    assert "Always pytest." in (await call("read_file", path=path)).result
    assert (await call("edit_file", path=path, old_text="Always", new_text="Only")).success
    assert (folder / "prefers-pytest.md").read_text().endswith("Only pytest.\n")
    assert "prefers-pytest.md (feedback)" in await Memory().index(sandbox)


async def test_a_memory_file_has_a_size_limit_and_other_files_do_not(sandbox, folder, call):
    big = "x" * (sandbox.layout.MEMORY_FILE_BYTES + 1)
    refused = await call("write_file", path=f"{folder}/big.md", content=big)
    assert not refused.success and "at most 100 kB; split it" in refused.error
    assert not (folder / "big.md").exists()
    assert (await call("write_file", path="big.txt", content=big)).success  # the workspace has none
    # An edit that would pass the limit is refused and the file stays as it was.
    assert (await call("write_file", path=f"{folder}/small.md", content="word\n")).success
    grown = await call("edit_file", path=f"{folder}/small.md", old_text="word", new_text=big)
    assert not grown.success and "at most 100 kB" in grown.error
    assert (folder / "small.md").read_text() == "word\n"
