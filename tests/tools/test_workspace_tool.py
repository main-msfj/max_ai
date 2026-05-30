from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.tools import WorkspaceTool
from max_ai.types.tool_call import ToolCallRecord


@pytest.mark.asyncio
async def test_workspace_tool_reads_and_lists_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    report = artifacts / "reports" / "summary.txt"
    report.parent.mkdir(parents=True)
    report.write_text("hello", encoding="utf-8")

    tool = WorkspaceTool()
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"artifacts_dir": str(artifacts)},
    )

    read = await tool.execute(
        ToolCallRecord(
            tool_name="workspace",
            parameters={"action": "read", "path": "reports/summary.txt"},
        ),
        context,
    )
    assert read.success is True
    assert read.result["content"] == "hello"

    listed = await tool.execute(
        ToolCallRecord(
            tool_name="workspace",
            parameters={"action": "list", "path": "reports"},
        ),
        context,
    )
    assert listed.success is True
    assert listed.result["items"] == [
        {"path": "reports/summary.txt", "type": "file", "bytes": 5}
    ]


@pytest.mark.asyncio
async def test_workspace_tool_rejects_paths_outside_artifacts(tmp_path: Path) -> None:
    tool = WorkspaceTool()
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"artifacts_dir": str(tmp_path / "artifacts")},
    )

    result = await tool.execute(
        ToolCallRecord(
            tool_name="workspace",
            parameters={"action": "read", "path": "../escape.txt"},
        ),
        context,
    )

    assert result.success is False
    assert "inside the artifacts directory" in (result.error or "")
