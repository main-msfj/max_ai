from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.base.tools import ToolContext
from max_ai.tools import WorkspaceTool
from max_ai.types.tool_call import ToolCallRecord


@pytest.mark.asyncio
async def test_workspace_tool_writes_reads_lists_and_deletes_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    tool = WorkspaceTool()
    context = ToolContext(
        run_id="run_1",
        user_id="u1",
        deps={"artifacts_dir": str(artifacts)},
    )

    write = await tool.execute(
        ToolCallRecord(
            tool_name="workspace",
            parameters={
                "action": "write",
                "path": "reports/summary.txt",
                "content": "hello",
            },
        ),
        context,
    )
    assert write.success is True
    assert write.result == {"path": "reports/summary.txt", "bytes": 5}

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

    deleted = await tool.execute(
        ToolCallRecord(
            tool_name="workspace",
            parameters={"action": "delete", "path": "reports/summary.txt"},
        ),
        context,
    )
    assert deleted.success is True
    assert deleted.result == {"path": "reports/summary.txt", "deleted": True}


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
            parameters={"action": "write", "path": "../escape.txt", "content": "no"},
        ),
        context,
    )

    assert result.success is False
    assert "inside the artifacts directory" in (result.error or "")
