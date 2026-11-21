"""ToolState: the tool calls of a run, kept by id."""

import pytest

from max_ai.core.tool.state import ToolState
from max_ai.types.tool_call import ToolCallRecord, ToolResult


def make_record(tool_name: str = "search") -> ToolCallRecord:
    return ToolCallRecord(tool_name=tool_name, parameters={})


def test_add_tracks_a_record_once():
    state = ToolState()
    assert state.records == {} and state.pending_user_input == []
    record = state.add(make_record())
    assert state.get(record.id) is record
    assert state.get("ghost") is None
    with pytest.raises(ValueError, match="already tracked"):
        state.add(record)


def test_a_question_waits_until_the_user_answers_it():
    state = ToolState()
    record = state.add(make_record("ask_user"))
    record.await_user_input()
    assert state.pending_user_input == [record]

    answer = ToolResult.success_result(record.id, {"answer": "blue"})
    record.store_result(answer)
    assert record.result == answer and state.pending_user_input == []


def test_a_call_left_running_by_a_dead_run_is_closed_as_failed():
    state = ToolState()
    record = state.add(make_record())
    record.auto_approve().start_execution()

    # As a session store brings it back: the record is still mid-execution.
    restored = ToolState.model_validate(state.model_dump())
    closed = restored.fail_stale(record.id, "the process died")
    assert closed.is_consumed and closed.result is not None and not closed.result.success
    with pytest.raises(ValueError):
        restored.fail_stale(record.id)
    with pytest.raises(KeyError):
        restored.fail_stale("ghost")
