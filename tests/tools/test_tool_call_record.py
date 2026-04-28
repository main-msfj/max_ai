"""Tests for ToolCallRecord — transitions, validations, properties."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.core.primitives import ToolCallStatus, FailureReason


def make_record(**overrides) -> ToolCallRecord:
    base = {"tool_name": "search_docs", "parameters": {"query": "x"}}
    base.update(overrides)
    return ToolCallRecord(**base)


# -------- BASIC SHAPE -----------------------------------------------------------
def test_required_fields():
    with pytest.raises(ValidationError):
        ToolCallRecord()  # type: ignore[call-arg]


def test_minimal_construction():
    r = make_record()
    assert r.tool_name == "search_docs"
    assert r.parameters == {"query": "x"}
    assert r.tool_type == "custom"
    assert r.status == ToolCallStatus.PENDING_APPROVAL
    assert r.id
    assert r.was_auto_approved is False
    assert r.started_at is None
    assert r.consumed_at is None
    assert r.result is None


def test_id_is_unique():
    assert make_record().id != make_record().id


def test_tool_type_validation():
    make_record(tool_type="mcp")
    make_record(tool_type="custom")
    make_record(tool_type="skill")
    with pytest.raises(ValidationError):
        make_record(tool_type="invalid")  # type: ignore[arg-type]


# -------- STATUS PROPERTIES -----------------------------------------------------------
def test_initial_state_is_pending():
    r = make_record()
    assert r.is_pending_approval
    assert not r.is_approved
    assert not r.is_rejected
    assert not r.is_executing
    assert not r.is_consumed
    assert not r.is_actionable
    assert not r.is_stale_execution


def test_approved_status_properties():
    r = make_record().approve()
    assert r.is_approved
    assert r.is_actionable
    assert not r.is_executing
    assert not r.is_consumed


def test_auto_approved_status_properties():
    r = make_record().auto_approve()
    assert r.is_approved
    assert r.is_actionable
    assert r.was_auto_approved


def test_executing_status_properties():
    r = make_record().approve().start_execution()
    assert r.is_executing
    assert r.is_stale_execution  # any executing is "stale" if seen across runs
    assert not r.is_actionable
    assert not r.is_consumed


def test_consumed_status_properties():
    r = make_record().approve().start_execution()
    result = ToolResult.success_result(tool_call_id=r.id, result="x")
    r.mark_consumed(result)
    assert r.is_consumed
    assert not r.is_executing
    assert not r.is_actionable


# -------- approve() -----------------------------------------------------------
def test_approve_returns_self():
    r = make_record()
    assert r.approve() is r


def test_approve_with_reason():
    r = make_record().approve(reason="user said go")
    assert r.status == ToolCallStatus.APPROVED
    assert r.approval_reason == "user said go"
    assert r.approval_decided_at is not None


def test_approve_only_from_pending():
    r = make_record().approve()
    with pytest.raises(ValueError, match="cannot be approved"):
        r.approve()


# -------- auto_approve() -----------------------------------------------------------
def test_auto_approve_sets_flag_and_status():
    r = make_record().auto_approve()
    assert r.status == ToolCallStatus.AUTO_APPROVED
    assert r.was_auto_approved is True
    assert r.approval_decided_at is not None


def test_auto_approve_only_from_pending():
    r = make_record().approve()
    with pytest.raises(ValueError, match="cannot be auto-approved"):
        r.auto_approve()


# -------- reject() -----------------------------------------------------------
def test_reject_default_reason():
    r = make_record().reject()
    assert r.status == ToolCallStatus.REJECTED
    assert r.approval_reason == "Rejected from UI."


def test_reject_custom_reason():
    r = make_record().reject(reason="wrong account")
    assert r.approval_reason == "wrong account"


def test_reject_only_from_pending():
    r = make_record().approve()
    with pytest.raises(ValueError, match="cannot be rejected"):
        r.reject()


# -------- start_execution() -----------------------------------------------------------
def test_start_execution_from_approved():
    r = make_record().approve()
    r.start_execution()
    assert r.is_executing
    assert r.started_at is not None


def test_start_execution_from_auto_approved():
    r = make_record().auto_approve()
    r.start_execution()
    assert r.is_executing


def test_start_execution_rejects_pending():
    r = make_record()
    with pytest.raises(ValueError, match="cannot start execution"):
        r.start_execution()


def test_start_execution_rejects_rejected():
    r = make_record().reject()
    with pytest.raises(ValueError, match="cannot start execution"):
        r.start_execution()


def test_start_execution_rejects_already_executing():
    r = make_record().approve().start_execution()
    with pytest.raises(ValueError, match="cannot start execution"):
        r.start_execution()


def test_start_execution_returns_self():
    r = make_record().approve()
    assert r.start_execution() is r


# -------- mark_consumed() -----------------------------------------------------------
def test_mark_consumed_from_executing():
    r = make_record().approve().start_execution()
    result = ToolResult.success_result(tool_call_id=r.id, result="done")
    r.mark_consumed(result)
    assert r.is_consumed
    assert r.consumed_at is not None
    assert r.result == result


def test_mark_consumed_from_auto_approved_executing():
    r = make_record().auto_approve().start_execution()
    result = ToolResult.success_result(tool_call_id=r.id, result="done")
    r.mark_consumed(result)
    assert r.is_consumed


def test_mark_consumed_rejects_pending():
    r = make_record()
    result = ToolResult.success_result(tool_call_id=r.id, result="x")
    with pytest.raises(ValueError, match="must be EXECUTING"):
        r.mark_consumed(result)


def test_mark_consumed_rejects_approved_not_started():
    r = make_record().approve()
    result = ToolResult.success_result(tool_call_id=r.id, result="x")
    with pytest.raises(ValueError, match="must be EXECUTING"):
        r.mark_consumed(result)


def test_mark_consumed_rejects_rejected():
    r = make_record().reject()
    result = ToolResult.success_result(tool_call_id=r.id, result="x")
    with pytest.raises(ValueError, match="must be EXECUTING"):
        r.mark_consumed(result)


def test_mark_consumed_rejects_already_consumed():
    r = make_record().auto_approve().start_execution()
    result = ToolResult.success_result(tool_call_id=r.id, result="x")
    r.mark_consumed(result)
    with pytest.raises(ValueError, match="already consumed"):
        r.mark_consumed(result)


def test_mark_consumed_rejects_mismatched_id():
    r = make_record().auto_approve().start_execution()
    result = ToolResult.success_result(tool_call_id="other", result="x")
    with pytest.raises(ValueError, match="does not match"):
        r.mark_consumed(result)


def test_mark_consumed_with_failure_result():
    """Failures (timeout, cancellation, runtime error) also reach CONSUMED."""
    r = make_record().auto_approve().start_execution()
    result = ToolResult.tool_failure(tool_call_id=r.id, error="boom")
    r.mark_consumed(result)
    assert r.is_consumed
    assert r.result is not None
    assert r.result.success is False


# -------- mark_stale_as_failed() -----------------------------------------------------------
def test_mark_stale_as_failed_default_reason():
    r = make_record().approve().start_execution()
    r.mark_stale_as_failed()
    assert r.is_consumed
    assert r.result is not None
    assert r.result.success is False
    assert r.result.failure_reason == FailureReason.EXECUTION_ERROR
    assert "did not complete" in (r.result.error or "")


def test_mark_stale_as_failed_custom_reason():
    r = make_record().auto_approve().start_execution()
    r.mark_stale_as_failed(reason="db connection lost during run")
    assert r.is_consumed
    assert r.result is not None
    assert r.result.error == "db connection lost during run"


def test_mark_stale_as_failed_rejects_pending():
    r = make_record()
    with pytest.raises(ValueError, match="not in a stale state"):
        r.mark_stale_as_failed()


def test_mark_stale_as_failed_rejects_consumed():
    r = make_record().auto_approve().start_execution()
    r.mark_consumed(ToolResult.success_result(tool_call_id=r.id, result="x"))
    with pytest.raises(ValueError, match="not in a stale state"):
        r.mark_stale_as_failed()


def test_mark_stale_as_failed_returns_self():
    r = make_record().approve().start_execution()
    assert r.mark_stale_as_failed() is r


# -------- CHAINING -----------------------------------------------------------
def test_full_chain_via_method_chaining():
    """Verify all transitions can be chained in one expression."""
    record = (
        ToolCallRecord(tool_name="search", parameters={"q": "x"})
        .auto_approve()
        .start_execution()
    )
    assert record.is_executing
    assert record.was_auto_approved


# -------- SERIALIZATION -----------------------------------------------------------
def test_round_trip_pending():
    r = make_record()
    restored = ToolCallRecord.model_validate(r.model_dump())
    assert restored == r


def test_round_trip_consumed():
    r = make_record().approve(reason="ok").start_execution()
    result = ToolResult.success_result(tool_call_id=r.id, result=[1, 2])
    r.mark_consumed(result)
    restored = ToolCallRecord.model_validate(r.model_dump())
    assert restored.id == r.id
    assert restored.status == ToolCallStatus.CONSUMED
    assert restored.result is not None
    assert restored.result.result == [1, 2]


def test_round_trip_executing_simulates_stale():
    """A serialized EXECUTING record (e.g., from a crashed run) rehydrates as stale."""
    r = make_record().approve().start_execution()
    restored = ToolCallRecord.model_validate(r.model_dump())
    assert restored.is_executing
    assert restored.is_stale_execution
    # Now we can recover it:
    restored.mark_stale_as_failed()
    assert restored.is_consumed
    assert restored.result is not None
    assert restored.result.success is False