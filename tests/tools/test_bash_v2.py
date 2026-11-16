"""bash runs commands in the sandbox and starts where the last one ended."""

from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.tools.bash_v2 import BashTool
from max_ai.core.tool import ToolDispatcher, ToolHook, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord

SKILL = "---\nname: hello\ndescription: Say hello\n---\nRun `python3 scripts/hello.py NAME`.\n"


@pytest.fixture
async def sandbox(tmp_path):
    """A user's sandbox with the ``hello`` skill, which has a script and a reference."""
    source = tmp_path / "source/hello"
    (source / "scripts").mkdir(parents=True)
    (source / "references").mkdir()
    (source / "SKILL.md").write_text(SKILL)
    (source / "scripts/hello.py").write_text("import sys\nprint('hello', sys.argv[1])\n")
    (source / "references/style.md").write_text("Always greet twice.\n")
    (tmp_path / "env").mkdir()
    bound = LocalSandbox(root=str(tmp_path / "env")).bind("user-1")
    await bound.prepare()
    await LocalSkills(str(tmp_path / "source"), ["hello"]).sync(bound)
    return bound


@pytest.fixture
def ctx():
    return RunContext(user_id="user-1", session_id="s1")


@pytest.fixture
def workspace(sandbox):
    return Path(sandbox.session_paths("s1")["workspace"])


class Approves(ToolHook):
    """Stands for the harness saying yes to every command."""

    async def permission(self, tool, record, tool_ctx):
        return True


async def bash(sandbox, ctx, command, hook=None, **more):
    parameters = {"command": command, "description": "test", **more}
    record = ToolCallRecord(tool_name="bash", parameters=parameters)
    dispatcher = ToolDispatcher(ToolRegistry([BashTool()]), sandbox, hook=hook or Approves())
    return await dispatcher.dispatch(record, ctx)


async def test_without_a_yes_the_command_waits_and_does_not_run(sandbox, ctx, workspace):
    assert await bash(sandbox, ctx, "touch ran.txt", hook=ToolHook()) is None
    assert not (workspace / "ran.txt").exists()


async def test_a_command_gives_its_exit_code_and_output(sandbox, ctx):
    assert (await bash(sandbox, ctx, "echo hi; echo oops >&2")).result == "exit code: 0\nhi\noops"
    failed = await bash(sandbox, ctx, "ls nothing-here")
    assert failed.success and failed.result.startswith("exit code: 2\n")
    assert (await bash(sandbox, ctx, "exit 7")).result == "exit code: 7"


async def test_it_starts_in_the_workspace_of_the_session(sandbox, ctx, workspace):
    assert (await bash(sandbox, ctx, "pwd -P")).result == f"exit code: 0\n{workspace.resolve()}"


async def test_the_next_command_starts_where_the_last_one_ended(sandbox, ctx, workspace):
    await bash(sandbox, ctx, "mkdir -p repo/src && cd repo")
    assert ctx.cwd == str(workspace.resolve() / "repo")
    assert (await bash(sandbox, ctx, "ls")).result == "exit code: 0\nsrc"
    await bash(sandbox, ctx, "cd ..")
    assert ctx.cwd is None  # back in the workspace


async def test_variables_do_not_pass_to_the_next_command(sandbox, ctx):
    await bash(sandbox, ctx, "export COLOR=red")
    assert (await bash(sandbox, ctx, 'echo "[$COLOR]"')).result == "exit code: 0\n[]"


async def test_a_folder_that_was_deleted_sends_it_back_to_the_workspace(sandbox, ctx, workspace):
    await bash(sandbox, ctx, "mkdir gone && cd gone")
    (workspace / "gone").rmdir()
    assert (await bash(sandbox, ctx, "pwd -P")).result.endswith(str(workspace.resolve()))


async def test_it_reads_the_files_of_a_skill_and_runs_its_script(sandbox, ctx):
    skill = f"{sandbox.layout.skills}/hello"
    seen = await bash(sandbox, ctx, f"cd {skill} && cat references/style.md")
    assert seen.result == "exit code: 0\nAlways greet twice."
    assert ctx.cwd.endswith("/skills/hello")  # it stays inside the skill
    assert (await bash(sandbox, ctx, "python3 scripts/hello.py marvin")).result == "exit code: 0\nhello marvin"


async def test_it_writes_a_python_program_and_runs_its_tests(sandbox, ctx, workspace):
    await bash(sandbox, ctx, "mkdir calc && cd calc && printf 'def add(a, b):\\n    return a + b\\n' > calc.py")
    test = "import unittest\\nfrom calc import add\\n\\nclass T(unittest.TestCase):\\n    def test_add(self):\\n        self.assertEqual(add(2, 3), 5)\\n"
    await bash(sandbox, ctx, f"printf '{test}' > test_calc.py")
    result = await bash(sandbox, ctx, "python3 -m unittest -q 2>&1 | tail -1")
    assert result.result == "exit code: 0\nOK"
    assert (workspace / "calc/test_calc.py").exists()


async def test_leaving_the_allowed_folders_sends_it_back(sandbox, ctx, workspace):
    left = await bash(sandbox, ctx, "cd /tmp && pwd")
    assert "[you left the allowed folders (/tmp); the next command starts in the workspace]" in left.result
    assert ctx.cwd is None
    assert (await bash(sandbox, ctx, "pwd -P")).result.endswith(str(workspace.resolve()))
    other = await bash(sandbox, ctx, "mkdir -p ../s2 && cd ../s2")  # another session
    assert "you left the allowed folders" in other.result and ctx.cwd is None


async def test_a_long_output_is_cut_in_the_middle_and_kept_in_a_file(sandbox, ctx):
    result = await bash(sandbox, ctx, "echo START; seq 1 20000; echo END")
    lines = result.result.splitlines()
    assert lines[1] == "START" and lines[-1] == "END" and len(result.result) < 31_000
    note = next(line for line in lines if "bytes cut" in line)
    log = Path(note.split("is in ")[1].removesuffix(" ...]"))
    assert log.read_text().count("\n") == 20_002
    short = await bash(sandbox, ctx, "echo hi")
    assert short.success and len(list(log.parent.iterdir())) == 1  # a short one leaves no file


async def test_a_command_that_takes_too_long_is_stopped(sandbox, ctx):
    result = await bash(sandbox, ctx, "echo before; sleep 30; echo after", timeout=1)
    assert result.result == "exit code: 124\nbefore\n[stopped after 1 seconds]"


async def test_the_output_cannot_fake_where_the_command_ended(sandbox, ctx):
    await bash(sandbox, ctx, "printf '__cwd_x__/etc'")
    assert ctx.cwd is None


async def test_an_empty_command_is_refused(sandbox, ctx):
    assert not (await bash(sandbox, ctx, "   ")).success
