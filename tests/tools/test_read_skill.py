"""read_skill returns the SKILL.md of a skill installed in the sandbox."""

from pathlib import Path

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
    (source / "scripts").mkdir()
    (source / "scripts/run.sh").write_text("echo hello\n")
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
    folder = sandbox.layout.skills / "hello"
    assert result.success
    assert result.result == f"Skill folder: {folder}\n  SKILL.md\n  scripts/run.sh\n\n{SKILL}"


async def test_a_skill_that_is_not_installed_says_so(sandbox):
    result = await read(sandbox, "missing")
    assert not result.success and "no skill named 'missing'" in result.error


async def test_a_skill_without_its_instructions_says_so(sandbox):
    Path(sandbox.layout.skills / "empty").mkdir()
    result = await read(sandbox, "empty")
    assert not result.success and "has no SKILL.md" in result.error


async def test_instructions_that_are_a_link_out_of_the_skill_are_not_read(sandbox, tmp_path):
    Path(sandbox.layout.skills / "linked").mkdir()
    Path(sandbox.layout.skills / "linked/SKILL.md").symlink_to(tmp_path / "env/secret.txt")
    result = await read(sandbox, "linked")
    assert not result.success and "not a skill" not in str(result.error)


@pytest.mark.parametrize("name", ["../secret.txt", "hello/../../secret.txt", "/etc/passwd", ""])
async def test_only_a_skill_name_is_accepted_never_a_path(sandbox, name):
    result = await read(sandbox, name)
    assert not result.success and "not a skill" not in (result.result or "")


async def test_without_a_sandbox_the_call_is_refused():
    result = await read(None, "hello")
    assert not result.success and "sandbox" in result.error
