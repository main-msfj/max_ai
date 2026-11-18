"""save_memory: the model gives a fact; the harness makes the file and its header."""

from pathlib import Path

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.stacks.builder import PromptBuilder
from max_ai.capabilities.stacks.task_analysis_layer import TaskAnalysisLayer
from max_ai.capabilities.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from max_ai.capabilities.tools.memory import SaveMemoryTool
from max_ai.core.harness.hooks import HarnessToolHook
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from tests.middleware.fakes import FakeClient, say

FACT = {
    "name": "prefers-pytest", "type": "feedback",
    "description": "Uses pytest, never unittest", "content": "Always write tests with pytest.",
}


@pytest.fixture
async def sandbox(tmp_path):
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    return bound


@pytest.fixture
def call(sandbox):
    """Call a tool as session ``s1``; the same conversation in every call."""
    ctx = RunContext(user_id="user-1", session_id="s1")
    tools = ToolRegistry([SaveMemoryTool(), ReadFileTool(), WriteFileTool(), EditFileTool()])

    async def run(tool, hook=None, **parameters):
        record = ToolCallRecord(tool_name=tool, parameters=parameters)
        return await ToolDispatcher(tools, sandbox, hook=hook).dispatch(record, ctx)

    return run


async def test_a_fact_becomes_a_file_with_its_header_and_a_line_of_the_index(sandbox, call):
    saved = await call("save_memory", **FACT)
    assert saved.success and saved.result == "Saved prefers-pytest.md in your memory."
    file = Path(str(sandbox.layout.memory)) / "prefers-pytest.md"
    assert file.read_text() == (
        "---\nname: prefers-pytest\ndescription: Uses pytest, never unittest\n"
        "type: feedback\n---\nAlways write tests with pytest.\n"
    )
    assert await Memory().index(sandbox) == "- prefers-pytest.md (feedback): Uses pytest, never unittest"


async def test_the_same_name_updates_the_fact_and_it_can_be_edited_after(sandbox, call):
    await call("save_memory", **FACT)
    again = await call("save_memory", **{**FACT, "content": "pytest, with fixtures."})
    assert again.success and again.result.startswith("Updated")
    file = Path(str(sandbox.layout.memory)) / "prefers-pytest.md"
    assert file.read_text().endswith("---\npytest, with fixtures.\n")
    assert len(list(file.parent.iterdir())) == 1
    # The model wrote it, so it may change a part of it without reading it first.
    edited = await call("edit_file", path=str(file), old_text="fixtures", new_text="plugins")
    assert edited.success, edited.error


@pytest.mark.parametrize(
    ("field", "value", "said"),
    [
        ("name", "../outside", "letters, digits and hyphens"),
        ("name", "two words", "letters, digits and hyphens"),
        ("type", "other", "type"),
        ("description", "two\nlines", "one line"),
        ("description", "x" * 151, "one line"),
        ("content", "  ", "content is empty"),
    ],
)
async def test_what_is_not_a_fact_is_refused_and_nothing_is_written(sandbox, call, field, value, said):
    refused = await call("save_memory", **{**FACT, field: value})
    assert not refused.success and said in refused.error
    assert list(Path(str(sandbox.layout.memory)).iterdir()) == []


async def test_a_fact_has_the_size_limit_of_a_memory_file(sandbox, call):
    big = "x" * (sandbox.layout.MEMORY_FILE_BYTES + 1)
    refused = await call("save_memory", **{**FACT, "content": big})
    assert not refused.success and "at most 100 kB" in refused.error


async def test_a_read_only_memory_has_no_tool_and_refuses_a_save(tmp_path, call):
    def tools(memory):
        made = Agent(
            "a", "d", "i", FakeClient(say("ok")), sandbox=LocalSandbox(str(tmp_path)), memory=memory
        )
        return [tool.name for tool in made.tools]

    # 1. The model only sees the tool when it may save.
    assert "save_memory" in tools(Memory())
    assert "save_memory" not in tools(Memory(read_only=True)) and "save_memory" not in tools(None)
    # 2. Called anyway, the harness refuses it.
    refused = await call("save_memory", hook=HarnessToolHook(memory=Memory(read_only=True)), **FACT)
    assert not refused.success and "read-only" in refused.error


async def test_the_prompt_says_when_to_save_only_when_it_can(sandbox):
    # 1. The memory's block names the tool and the moments.
    block = Memory().block(sandbox, "")
    assert "Save a fact with save_memory when" in block and "the user corrects you" in block
    assert "save_memory" not in Memory(read_only=True).block(sandbox, "")
    # 2. The layer that says how to work reminds it before the final answer.
    line = "save_memory first"
    assert line in PromptBuilder([TaskAnalysisLayer()], ToolRegistry([SaveMemoryTool()])).build()
    assert line not in PromptBuilder([TaskAnalysisLayer()], ToolRegistry()).build()
