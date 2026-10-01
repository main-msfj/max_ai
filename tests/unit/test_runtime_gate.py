"""RuntimeCompletionGate: intermediate files a turn leaves in the workspace
get one nudge, then only a note. It reads the bash results, never the disk."""

from __future__ import annotations

from max_ai.capabilities.completion_gate import RuntimeCompletionGate
from max_ai.core.messages import AssistantMessage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult


def closing_after(*calls: tuple[str, dict]) -> RunContext:
    """A turn that ran ``calls`` (tool name, result data) and then answered."""
    ctx = RunContext(messages=[AssistantMessage(source="a", content="Done.")])
    for name, data in calls:
        record = ToolCallRecord(tool_name=name, parameters={"command": "x"})
        ctx.tool_state.add(record.auto_approve().start_execution())
        ctx.tool_state.consume(record.id, ToolResult.success_result(record.id, data))
    return ctx


def bash(**files: list[str]) -> tuple[str, dict]:
    return "bash", {"exit_code": 0, "output": "", "files": files}


def test_files_left_in_tmp_are_nudged_once_then_noted():
    gate = RuntimeCompletionGate()
    ctx = closing_after(bash(created=["report.xlsx", "A/tmp/s.py"]))
    first = gate.on_final_response(ctx)
    assert first.status == "incomplete"
    assert "A/tmp/s.py" in first.reasons[0] and "report.xlsx" not in first.reasons[0]
    again = gate.on_final_response(ctx)  # the model closes without cleaning
    assert again.status == "completed" and "A/tmp/s.py" in again.reasons[0]


def test_files_deleted_later_in_the_turn_are_not_left():
    gate = RuntimeCompletionGate()
    by_rm = closing_after(bash(created=["tmp/a.py"]), bash(deleted=["tmp/a.py"]))
    by_tool = closing_after(bash(created=["tmp/a.py"]), ("DeleteFile", {"path": "workspace/tmp/a.py"}))
    assert gate.on_final_response(by_rm).status == "completed"
    assert gate.on_final_response(by_tool).status == "completed"


def test_deliverable_folders_are_not_intermediate():
    ctx = closing_after(bash(created=["output/spreadsheet/plan.xlsx", "tmpfile.txt"]))
    assert RuntimeCompletionGate().on_final_response(ctx).status == "completed"
