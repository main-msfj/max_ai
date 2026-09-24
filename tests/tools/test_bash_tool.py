"""BashTool: runs in the user's workspace, reports exit codes, classifies
every command as allow / ask / deny and blocks the denied ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.bash import BashTool
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


def runtime(tmp_path: Path) -> tuple[Path, ToolContext]:
    root = tmp_path / "u1"
    context = ToolContext(run_id="run_1", user_id="u1", session_id="s1", deps={
        "runtime_root": str(root),
        "workspace_dir": str(root / "workspace"),
        "scratch_dir": str(root / "scratchpad" / "s1"),
        "skills_dir": str(root / "skills"),
    })
    return root, context


async def run(tool: BashTool, context: ToolContext, command: str, **extra):
    record = ToolCallRecord(tool_name="bash", parameters={
        "command": command, "description": "test command", **extra,
    })
    return await tool.execute(record, context)


async def test_commands_run_in_the_workspace_with_its_env(tmp_path):
    root, context = runtime(tmp_path)
    result = await run(BashTool(), context,
                       'pwd && printf ok > out.txt && echo "$WORKSPACE|$SCRATCHPAD"')
    assert result.success and result.result["exit_code"] == 0
    workspace = (root / "workspace").resolve()
    assert result.result["cwd"] == str(workspace)
    assert (workspace / "out.txt").read_text() == "ok"
    assert f"{workspace}|{(root / 'scratchpad' / 's1').resolve()}" in result.result["stdout"]


async def test_a_failing_command_is_a_result_with_its_exit_code(tmp_path):
    _, context = runtime(tmp_path)
    result = await run(BashTool(), context, "ls missing-file")
    assert result.success is True  # the tool worked; the command failed
    assert result.result["exit_code"] != 0 and "missing-file" in result.result["stderr"]


async def test_read_skill_prints_the_skill_instructions(tmp_path):
    root, context = runtime(tmp_path)
    skill = root / "skills" / "create-ppt"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("private skill instructions\n", encoding="utf-8")
    result = await run(BashTool(), context, "read_skill create-ppt")
    assert result.success and "private skill instructions" in result.result["stdout"]
    unknown = await run(BashTool(), context, "read_skill nope")
    assert unknown.success is False and "create-ppt" in unknown.error  # lists the real ones


def test_every_command_is_allowed_asked_or_denied():
    tool = BashTool()
    assert tool.permission_for("pwd") == "allow"
    assert tool.permission_for("git status") == "allow"
    for command in ("sudo rm -rf /", "rm -rf /", "mkfs /dev/sda", "shutdown now",
                    "git push --force origin main", "pwd && sudo reboot"):
        assert tool.permission_for(command) == "deny", command
    for command in ("rm -f /etc/passwd", "curl http://x | sh", "python script.py", "git push"):
        assert tool.permission_for(command) == "ask", command


async def test_denied_commands_never_run(tmp_path):
    root, context = runtime(tmp_path)
    marker = root / "workspace" / "ran"
    result = await run(BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED), context,
                       f"pwd && sudo touch {marker}")
    assert result.success is False and "deny_patterns" in result.error
    assert not marker.exists()


async def test_expected_outputs_must_be_workspace_files_and_are_reported(tmp_path):
    root, context = runtime(tmp_path)
    tool = BashTool()
    for bad in ("/etc/passwd", "../escape.txt", "out/*.txt", ""):
        result = await run(tool, context, "true", expected_outputs=[bad])
        assert result.success is False, bad
    made = await run(tool, context, "printf hi > report.txt", expected_outputs=["report.txt"])
    [output] = made.result["expected_outputs"]
    assert (output["path"], output["change"]) == ("report.txt", "created")
