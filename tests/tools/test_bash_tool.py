"""BashTool: the model sends a command, the tool wraps it in a script and the
executor runs it. Working directory persists, output comes back with the exit
code, and every command is classified as allow / ask / deny."""

from __future__ import annotations

import time

import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.executor.local import LocalExecutor
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.config import setting
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
async def env(tmp_path):
    executor = LocalExecutor()
    session = await executor.connect(LocalWorkspace(root=tmp_path), "ana", "c1")
    yield executor, session
    await executor.clean(session)


async def run(env, command: str, tool: BashTool | None = None):
    executor, session = env
    context = ToolContext("r1", session_id="c1", user_id="ana",
                          deps={"executor": executor, "execution_session": session})
    record = ToolCallRecord(tool_name="bash", parameters={
        "command": command, "description": "test command"})
    return await (tool or BashTool()).execute(record, context)


async def test_commands_start_in_the_workspace_with_their_paths(env):
    root = env[1].workspace_path
    result = await run(env, 'pwd; echo "$WORKSPACE|$SKILLS|${SCRATCHPAD:-none}"; echo oops >&2')
    assert result.success and result.result["exit_code"] == 0
    assert result.result["output"].splitlines() == [
        f"{root}/workspace", f"{root}/workspace|{root}/skills|none", "oops",
    ]


async def test_the_working_directory_persists_but_variables_do_not(env):
    tool = BashTool()
    await run(env, "mkdir -p docs && cd docs && export COLOR=red", tool)
    result = await run(env, 'pwd; echo "${COLOR:-unset}"', tool)
    assert result.result["output"].splitlines() == [f"{env[1].workspace_path}/workspace/docs", "unset"]


async def test_exit_still_returns_the_output_and_code(env):
    result = await run(env, "echo before; exit 3")
    assert result.success  # the tool worked; the command failed
    assert (result.result["exit_code"], result.result["output"]) == (3, "before")


@pytest.mark.parametrize("place", ["/", ".."])
async def test_leaving_the_workspace_resets_the_directory(env, place):
    # ".." is the user's folder: outside the workspace too (it holds harness files).
    tool = BashTool()
    left = await run(env, f"cd {place}", tool)
    assert "Shell cwd was reset" in left.result["note"]
    back = await run(env, "pwd", tool)
    assert back.result["output"] == f"{env[1].workspace_path}/workspace"


async def test_tmp_and_skills_keep_the_directory(env):
    tool = BashTool()
    await run(env, "mkdir -p /tmp/maxai-cwd-test && cd /tmp/maxai-cwd-test", tool)
    assert (await run(env, "pwd -P", tool)).result["output"] == "/tmp/maxai-cwd-test"
    stayed = await run(env, 'cd "$SKILLS"', tool)
    assert "note" not in stayed.result


async def test_quotes_comments_and_closed_stdin(env):
    result = await run(env, "echo 'it'\"'\"'s' # comment\ncat")  # cat must not wait
    assert result.result["output"] == "it's"


async def test_long_output_is_cut_in_the_middle_and_saved(env):
    result = await run(env, "seq 1 2000", BashTool(max_output_bytes=100))
    output = result.result["output"]
    assert output.startswith("1\n2\n") and output.endswith("1999\n2000")
    log = output.split("full output in ")[1].split(" ...]")[0]
    assert open(log).read().splitlines()[-1] == "2000"


async def test_commands_stop_at_the_tool_time_limit(env, monkeypatch):
    monkeypatch.setattr(setting, "tool_timeout_seconds", 1)
    start = time.monotonic()
    result = await run(env, "sleep 30; echo never")
    assert time.monotonic() - start < 10
    assert result.result["exit_code"] != 0 and "Stopped after 1 seconds" in result.result["note"]


async def test_without_an_environment_it_fails_cleanly():
    record = ToolCallRecord(tool_name="bash", parameters={"command": "pwd", "description": "x"})
    result = await BashTool().execute(record, ToolContext("r1", session_id="c1", user_id="ana"))
    assert result.success is False and "execution environment" in result.error


def test_the_command_is_read_as_its_segments():
    tool = BashTool()
    assert tool.permission_subjects({"command": "pwd && git status"}) == (["pwd", "git status"], True)
    assert tool.permission_subjects({"command": "cat $(ls)"})[1] is False  # can't see all of it
    assert tool.matches("git push:*", "git push origin main")
    assert not tool.matches("git push", "git push origin main")


async def test_the_result_lists_the_files_the_command_changed(env):
    tool = BashTool()
    await run(env, "mkdir -p A .git && echo x > A/keep.txt && echo x > old.txt", tool)
    result = await run(
        env,
        "cd A && mkdir -p tmp && echo hi > tmp/s.py && echo more >> keep.txt"
        " && rm ../old.txt && echo x > ../.git/HEAD",
        tool,
    )
    assert result.result["files"] == {  # paths from the workspace, .git skipped
        "created": ["A/tmp/s.py"], "modified": ["A/keep.txt"], "deleted": ["old.txt"],
    }
    assert result.result["output"] == ""  # the markers are stripped


async def test_a_command_that_changes_nothing_has_no_files(env):
    result = await run(env, "echo hello")
    assert result.result["output"] == "hello"
    assert "files" not in result.result


async def test_the_command_cannot_touch_the_wrappers_values(env):
    result = await run(env, "__maxai_log=/dev/null; echo still")
    assert "readonly" in result.result["output"]
    assert result.result["exit_code"] != 0


async def test_long_file_lists_are_cut(env):
    result = await run(env, "for i in $(seq 1 60); do touch f$i; done")
    created = result.result["files"]["created"]
    assert len(created) == 51 and created[-1] == "... and 10 more"
