"""Tests for ToolState — container operations."""

from __future__ import annotations

import pytest

from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.core.tool_state import ToolState


# -------- HELPERS -----------------------------------------------------------
def make_record(tool_name: str = "search", **overrides) -> ToolCallRecord:
    base = {"tool_name": tool_name, "parameters": {}}
    base.update(overrides)
    return ToolCallRecord(**base)


# -------- BASIC SHAPE -----------------------------------------------------------
def test_empty_state():
    state = ToolState()
    assert state.records == {}
    assert not state.waiting_for_approval
    assert state.pending_approvals == []
    assert state.actionable_calls == []
    assert state.rejected_calls == []
    assert state.consumed_calls == []


# -------- add() -----------------------------------------------------------
def test_add_record():
    state = ToolState()
    record = make_record()
    state.add(record)
    assert state.has(record.id)
    assert state.get(record.id) is record


def test_add_duplicate_raises():
    state = ToolState()
    record = make_record()
    state.add(record)
    with pytest.raises(ValueError, match="already tracked"):
        state.add(record)


def test_get_returns_none_for_unknown():
    state = ToolState()
    assert state.get("nonexistent") is None
    assert not state.has("nonexistent")


# -------- waiting_for_approval -----------------------------------------------------------
def test_waiting_for_approval_with_pending():
    state = ToolState()
    state.add(make_record())
    assert state.waiting_for_approval


def test_waiting_for_approval_after_decision():
    """After all pending records are approved/rejected, waiting=False."""
    state = ToolState()
    r1 = make_record()
    r2 = make_record()
    state.add(r1)
    state.add(r2)
    assert state.waiting_for_approval

    state.apply_approval(r1.id, approved=True)
    assert state.waiting_for_approval  # r2 still pending

    state.apply_approval(r2.id, approved=False)
    assert not state.waiting_for_approval


# -------- LIST QUERIES -----------------------------------------------------------
def test_pending_approvals_only_returns_pending():
    state = ToolState()
    pending = make_record("pending_one")
    approved = make_record("approved_one")
    rejected = make_record("rejected_one")
    state.add(pending)
    state.add(approved)
    state.add(rejected)
    state.apply_approval(approved.id, approved=True)
    state.apply_approval(rejected.id, approved=False)

    pendings = state.pending_approvals
    assert len(pendings) == 1
    assert pendings[0].id == pending.id


def test_actionable_calls_returns_approved_and_auto_approved():
    state = ToolState()
    pending = make_record("p")
    approved = make_record("a")
    auto = make_record("ato")
    rejected = make_record("r")
    state.add(pending)
    state.add(approved)
    state.add(auto)
    state.add(rejected)
    state.apply_approval(approved.id, approved=True)
    auto.auto_approve()
    state.apply_approval(rejected.id, approved=False)

    actionable_ids = {r.id for r in state.actionable_calls}
    assert actionable_ids == {approved.id, auto.id}


def test_actionable_excludes_consumed():
    state = ToolState()
    record = make_record()
    state.add(record)
    state.apply_approval(record.id, approved=True)
    assert record in state.actionable_calls

    result = ToolResult.success_result(tool_call_id=record.id, result="ok")
    state.consume(record.id, result)
    assert record not in state.actionable_calls
    assert record in state.consumed_calls


def test_rejected_calls():
    state = ToolState()
    r1 = make_record("a")
    r2 = make_record("b")
    state.add(r1)
    state.add(r2)
    state.apply_approval(r1.id, approved=False, reason="bad")
    state.apply_approval(r2.id, approved=True)

    rejecteds = state.rejected_calls
    assert len(rejecteds) == 1
    assert rejecteds[0].id == r1.id
    assert rejecteds[0].approval_reason == "bad"


# -------- apply_approval() -----------------------------------------------------------
def test_apply_approval_approve():
    state = ToolState()
    record = make_record()
    state.add(record)
    returned = state.apply_approval(record.id, approved=True, reason="go")
    assert returned is record
    assert record.is_approved
    assert record.approval_reason == "go"


def test_apply_approval_reject():
    state = ToolState()
    record = make_record()
    state.add(record)
    state.apply_approval(record.id, approved=False, reason="no")
    assert record.is_rejected
    assert record.approval_reason == "no"


def test_apply_approval_unknown_id_raises():
    state = ToolState()
    with pytest.raises(KeyError, match="no tool call"):
        state.apply_approval("ghost", approved=True)


# -------- consume() -----------------------------------------------------------
def test_consume_approved_record():
    state = ToolState()
    record = make_record()
    state.add(record)
    state.apply_approval(record.id, approved=True)

    result = ToolResult.success_result(tool_call_id=record.id, result=42)
    returned = state.consume(record.id, result)

    assert returned is record
    assert record.is_consumed
    assert record.result == result
    assert state.get_result(record.id) == result


def test_consume_unknown_id_raises():
    state = ToolState()
    result = ToolResult.success_result(tool_call_id="ghost", result=None)
    with pytest.raises(KeyError, match="no tool call"):
        state.consume("ghost", result)


def test_consume_pending_raises():
    """Pending records can't be consumed (must approve first)."""
    state = ToolState()
    record = make_record()
    state.add(record)
    result = ToolResult.success_result(tool_call_id=record.id, result=None)
    with pytest.raises(ValueError, match="cannot be consumed"):
        state.consume(record.id, result)


def test_consume_already_consumed_raises():
    state = ToolState()
    record = make_record()
    state.add(record)
    record.auto_approve()
    result = ToolResult.success_result(tool_call_id=record.id, result=None)
    state.consume(record.id, result)
    with pytest.raises(ValueError, match="already consumed"):
        state.consume(record.id, result)


# -------- get_result() -----------------------------------------------------------
def test_get_result_before_consumed():
    state = ToolState()
    record = make_record()
    state.add(record)
    assert state.get_result(record.id) is None


def test_get_result_after_consumed():
    state = ToolState()
    record = make_record()
    state.add(record)
    record.auto_approve()
    result = ToolResult.success_result(tool_call_id=record.id, result="x")
    state.consume(record.id, result)
    assert state.get_result(record.id) == result


def test_get_result_unknown_id():
    state = ToolState()
    assert state.get_result("ghost") is None


# -------- reset() -----------------------------------------------------------
def test_reset_clears_records():
    state = ToolState()
    state.add(make_record())
    state.add(make_record())
    state.reset()
    assert state.records == {}


# -------- load_from() -----------------------------------------------------------
def test_load_from_empty():
    state = ToolState.load_from()
    assert state.records == {}


def test_load_from_serialized():
    original = ToolState()
    record = make_record()
    original.add(record)
    record.approve(reason="ok")

    dumped = original.model_dump()
    restored = ToolState.load_from(dumped)
    assert restored.has(record.id)
    assert restored.get(record.id).is_approved  # type: ignore[union-attr]


def test_load_from_with_approval_applied():
    """Resume scenario: load state and apply pending approval in one step."""
    original = ToolState()
    record = make_record()
    original.add(record)
    dumped = original.model_dump()

    restored = ToolState.load_from(
        dumped,
        approval=(record.id, True, "user said yes"),
    )
    restored_record = restored.get(record.id)
    assert restored_record is not None
    assert restored_record.is_approved
    assert restored_record.approval_reason == "user said yes"


def test_load_from_with_unknown_approval_raises():
    """Resume with approval for a record that wasn't in the state."""
    original = ToolState()
    dumped = original.model_dump()
    with pytest.raises(KeyError):
        ToolState.load_from(dumped, approval=("ghost", True, None))


# -------- SERIALIZATION ROUND-TRIP -----------------------------------------------------------
def test_full_lifecycle_round_trip():
    """Build a state with all 3 lifecycle stages, serialize, restore, verify."""
    state = ToolState()

    # consumed
    consumed = make_record("search_docs", parameters={"q": "abc"})
    state.add(consumed)
    consumed.auto_approve()
    state.consume(
        consumed.id,
        ToolResult.success_result(tool_call_id=consumed.id, result=["doc1"]),
    )

    # rejected
    rejected = make_record("delete_user")
    state.add(rejected)
    state.apply_approval(rejected.id, approved=False, reason="wrong account")

    # pending
    pending = make_record("send_email")
    state.add(pending)

    dumped = state.model_dump()
    restored = ToolState.load_from(dumped)

    assert len(restored.records) == 3
    assert restored.get(consumed.id).is_consumed  # type: ignore[union-attr]
    assert restored.get(rejected.id).is_rejected  # type: ignore[union-attr]
    assert restored.get(pending.id).is_pending_approval  # type: ignore[union-attr]

# -------- EXECUTING + STALE -----------------------------------------------------------
def test_executing_calls_query():
    state = ToolState()
    r1 = make_record("a")
    r2 = make_record("b")
    state.add(r1).auto_approve().start_execution()
    state.add(r2)  # stays pending

    executing = state.executing_calls
    assert len(executing) == 1
    assert executing[0].id == r1.id


def test_stale_executions_query():
    """A serialized state with EXECUTING records → those are stale."""
    state = ToolState()
    r = make_record()
    state.add(r).approve().start_execution()

    # Simulate persistence + rehydration
    dumped = state.model_dump()
    restored = ToolState.load_from(dumped)

    stale = restored.stale_executions
    assert len(stale) == 1
    assert stale[0].id == r.id


def test_fail_stale_recovers_record():
    state = ToolState()
    r = make_record()
    state.add(r).approve().start_execution()

    state.fail_stale(r.id, reason="prev run crashed")

    record = state.get(r.id)
    assert record is not None
    assert record.is_consumed
    assert record.result is not None
    assert record.result.success is False
    assert record.result.error == "prev run crashed"


def test_fail_stale_unknown_id_raises():
    state = ToolState()
    with pytest.raises(KeyError):
        state.fail_stale("ghost")


def test_fail_stale_rejects_non_stale():
    state = ToolState()
    r = make_record()
    state.add(r)
    with pytest.raises(ValueError, match="not in a stale state"):
        state.fail_stale(r.id)


# -------- FULL LIFECYCLE WITH CRASH RECOVERY -----------------------------------------------------------
def test_crash_recovery_workflow():
    """End-to-end: state with a crashed tool, agent recovers it on rehydrate."""
    # Run #1: tool starts executing, process dies
    state_run1 = ToolState()
    r = make_record("delete_user")
    state_run1.add(r).auto_approve().start_execution()

    serialized = state_run1.model_dump()

    # Run #2: rehydrate, find stale, mark failed
    state_run2 = ToolState.load_from(serialized)
    assert len(state_run2.stale_executions) == 1

    for stale in state_run2.stale_executions:
        state_run2.fail_stale(stale.id)

    # After recovery: no more executing, one consumed (failed)
    assert state_run2.executing_calls == []
    assert state_run2.stale_executions == []
    assert len(state_run2.consumed_calls) == 1
    assert state_run2.consumed_calls[0].result.success is False  # type: ignore[union-attr]