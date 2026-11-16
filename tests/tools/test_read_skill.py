"""read_skill returns the SKILL.md of a skill installed in the sandbox."""

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.tools.skill import ReadSkillTool
from max_ai.core.tool import ToolDispatcher, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

SKILL = "---\nname: hello\ndescription: Say hello from the sandbox\n---\nRun `run.sh` and report what it prints.\n"


@pytest.fixture
async def sandbox(tmp_path):
    """A user's sandbox with the ``hello`` skill installed."""
    source = tmp_path / "source/hello"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(SKILL)
    (tmp_path / "env").mkdir()
    (tmp_path / "env/secret.txt").write_text("not a skill")
    bound = LocalSandbox(root=str(tmp_path / "env")).bind("user-1")
    await bound.prepare()
    await LocalSkills(str(tmp_path / "source"), ["hello"]).sync(bound)
    return bound


async def read(sandbox, skill_name):
    record = ToolCallRecord(tool_name="read_skill", parameters={"skill_name": skill_name})
    ctx = RunContext(user_id="user-1", session_id="s1")
    return await ToolDispatcher(ToolRegistry([ReadSkillTool()]), sandbox).dispatch(record, ctx)


async def test_it_returns_the_instructions_of_the_skill(sandbox):
    result = await read(sandbox, "hello")
    assert result.success and result.result == SKILL


async def test_a_skill_that_is_not_installed_says_so(sandbox):
    result = await read(sandbox, "missing")
    assert not result.success and "no skill named 'missing'" in result.error


@pytest.mark.parametrize("name", ["../secret.txt", "hello/../../secret.txt", "/etc/passwd", ""])
async def test_only_a_skill_name_is_accepted_never_a_path(sandbox, name):
    result = await read(sandbox, name)
    assert not result.success and "not a skill" not in (result.result or "")


async def test_without_a_sandbox_the_call_is_refused():
    result = await read(None, "hello")
    assert not result.success and "sandbox" in result.error
