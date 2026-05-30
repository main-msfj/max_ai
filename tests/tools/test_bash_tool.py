from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.tools import BashTool
from max_ai.types.tool_call import ToolCallRecord
from max_ai.types.tools import ToolApprovalMode


@pytest.mark.asyncio
async def test_bash_tool_runs_from_runtime_root(tmp_path: Path) -> None:
    root = tmp_path / "tmp" / "u1"
    tools = root / "tools"
    skills = root / "skills"
    artifacts = root / "artifacts"
    skills.mkdir(parents=True)
    tools.mkdir()
    artifacts.mkdir()

    tool = BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED)
    record = ToolCallRecord(
        tool_name="bash",
        parameters={"command": "pwd && ls && printf test > artifacts/out.txt"},
    )
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={
            "runtime_root": str(root),
            "tools_dir": str(tools),
            "skills_dir": str(skills),
            "artifacts_dir": str(artifacts),
        },
    )

    result = await tool.execute(record, context)

    assert result.success is True
    assert result.result["exit_code"] == 0
    assert result.result["cwd"] == str(root.resolve())
    assert str(root.resolve()) in result.result["stdout"]
    assert "skills" in result.result["stdout"]
    assert "tools" in result.result["stdout"]
    assert "artifacts" in result.result["stdout"]
    assert (artifacts / "out.txt").read_text(encoding="utf-8") == "test"


@pytest.mark.asyncio
async def test_bash_tool_returns_nonzero_exit_code_as_output(tmp_path: Path) -> None:
    root = tmp_path / "tmp" / "u1"
    tool = BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED)
    record = ToolCallRecord(
        tool_name="bash",
        parameters={"command": "ls missing-file"},
    )
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"runtime_root": str(root)},
    )

    result = await tool.execute(record, context)

    assert result.success is True
    assert result.result["exit_code"] != 0
    assert "missing-file" in result.result["stderr"]


@pytest.mark.asyncio
async def test_bash_tool_read_skill_alias_reads_skill_instructions(tmp_path: Path) -> None:
    root = tmp_path / "tmp" / "u1"
    skill_dir = root / "skills" / "create-ppt"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("private skill instructions\n", encoding="utf-8")

    tool = BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED)
    record = ToolCallRecord(
        tool_name="bash",
        parameters={"command": "read_skill create-ppt"},
    )
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"runtime_root": str(root)},
    )

    result = await tool.execute(record, context)

    assert result.success is True
    assert result.result["exit_code"] == 0
    assert "private skill instructions" in result.result["stdout"]


@pytest.mark.asyncio
async def test_bash_tool_blocks_destructive_shell_commands(tmp_path: Path) -> None:
    root = tmp_path / "tmp" / "u1"
    tool = BashTool(approval_mode=ToolApprovalMode.AUTO_APPROVED)
    record = ToolCallRecord(
        tool_name="bash",
        parameters={"command": "rm -f /etc/passwd"},
    )
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"runtime_root": str(root)},
    )

    result = await tool.execute(record, context)

    assert result.success is False
    assert result.error is not None
    assert "Command blocked: destructive operation" in result.error
