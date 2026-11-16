"""ToolState: the tool calls of a run, and what the application does to them."""

import pytest

from max_ai.core.tool.state import ToolState
from max_ai.types.tool_call import ToolCallRecord, ToolResult


def make_record(tool_name: str = "search") -> ToolCallRecord:
    return ToolCallRecord(tool_name=tool_name, parameters={})


def test_an_empty_state_waits_for_nothing():
    state = ToolState()
    assert state.records == {}
    assert not state.waiting_for_approval and not state.waiting_for_input
    assert state.pending_approvals == state.pending_user_input == state.stale_executions == []


def test_add_tracks_a_record_once():
    state = ToolState()
    record = state.add(make_record())
    assert state.get(record.id) is record
    assert state.get("ghost") is None
    with pytest.raises(ValueError, match="already tracked"):
        state.add(record)


def test_a_new_record_waits_for_approval_until_the_user_decides():
    state = ToolState()
    yes, no = state.add(make_record()), state.add(make_record())
    assert state.waiting_for_approval and state.pending_approvals == [yes, no]

    state.apply_approval(yes.id, True, "go ahead")
    state.apply_approval(no.id, False, "not that one")
    assert yes.is_actionable and no.is_rejected
    assert no.approval_reason == "not that one"
    assert not state.waiting_for_approval


def test_an_approval_needs_a_tracked_record_that_still_waits():
    state = ToolState()
    with pytest.raises(KeyError):
        state.apply_approval("ghost", True)
    record = state.add(make_record())
    state.apply_approval(record.id, True)
    with pytest.raises(ValueError):
        state.apply_approval(record.id, True)


def test_a_question_waits_until_the_user_answers_it():
    state = ToolState()
    record = state.add(make_record("ask_user"))
    record.await_user_input()
    assert state.waiting_for_input and state.pending_user_input == [record]

    answer = ToolResult.success_result(record.id, {"answer": "blue"})
    state.provide_result(record.id, answer)
    assert record.result == answer and not state.waiting_for_input
    with pytest.raises(KeyError):
        state.provide_result("ghost", answer)


def test_a_call_left_running_by_a_dead_run_is_found_and_closed_as_failed():
    state = ToolState()
    record = state.add(make_record())
    record.auto_approve().start_execution()

    restored = ToolState.load_from(state.model_dump())
    assert [r.id for r in restored.stale_executions] == [record.id]
    closed = restored.fail_stale(record.id, "the process died")
    assert closed.is_consumed and closed.result is not None and not closed.result.success
    assert restored.stale_executions == []
    with pytest.raises(ValueError):
        restored.fail_stale(record.id)


def test_load_from_restores_the_records_or_starts_empty():
    state = ToolState()
    record = state.add(make_record())
    record.auto_approve().start_execution()
    record.mark_consumed(ToolResult.success_result(record.id, {"n": 1}))

    restored = ToolState.load_from(state.model_dump())
    kept = restored.get(record.id)
    assert kept is not None and kept.is_consumed and kept.result.result == {"n": 1}
    assert ToolState.load_from(None).records == {}


def test_reset_forgets_every_call():
    state = ToolState()
    state.add(make_record())
    state.reset()
    assert state.records == {}
