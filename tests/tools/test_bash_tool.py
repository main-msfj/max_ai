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
from max_ai.types.tools import ToolApprovalMode


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
    result = await run(env, 'pwd; echo "$SCRATCHPAD|$SKILLS"; echo oops >&2')
    assert result.success and result.result["exit_code"] == 0
    assert result.result["output"].splitlines() == [
        f"{root}/workspace", f"{root}/scratchpad/c1|{root}/skills", "oops",
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


async def test_leaving_the_users_files_resets_the_directory(env):
    tool = BashTool()
    left = await run(env, "cd /", tool)
    assert "outside the user's files" in left.result["note"]
    back = await run(env, "pwd", tool)
    assert back.result["output"] == f"{env[1].workspace_path}/workspace"


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


def test_every_command_is_allowed_asked_or_denied():
    tool = BashTool()
    assert tool.permission_for("pwd") == "allow"
    assert tool.permission_for("git status") == "allow"
    for command in ("sudo rm -rf /", "rm -rf /", "mkfs /dev/sda", "shutdown now",
                    "git push --force origin main", "pwd && sudo reboot"):
        assert tool.permission_for(command) == "deny", command
    for command in ("rm -f /etc/passwd", "curl http://x | sh", "python script.py", "git push"):
        assert tool.permission_for(command) == "ask", command


async def test_denied_commands_never_run(env):
    marker = f"{env[1].workspace_path}/workspace/ran"
    result = await run(env, f"pwd && sudo touch {marker}",
                       BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED))
    assert result.success is False and "deny_patterns" in result.error
