"""Tests for ToolResult — factories, duration, immutability."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from max_ai.types.tool_call import ToolResult
from max_ai.core.primitives import FailureReason


# -------- BASIC SHAPE -----------------------------------------------------------
def test_required_fields():
    """tool_call_id and success are required."""
    with pytest.raises(ValidationError):
        ToolResult()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ToolResult(success=True)  # type: ignore[call-arg]


def test_minimal_construction():
    r = ToolResult(success=True, tool_call_id="abc123")
    assert r.success is True
    assert r.tool_call_id == "abc123"
    assert r.result is None
    assert r.error is None
    assert r.failure_reason is None
    assert r.metadata == {}
    assert r.completed_at is None
    assert isinstance(r.started_at, datetime)


def test_is_frozen():
    """ToolResult is immutable."""
    r = ToolResult(success=True, tool_call_id="abc")
    with pytest.raises(ValidationError):
        r.success = False  # type: ignore[misc]


# -------- DURATION -----------------------------------------------------------
def test_duration_ms_when_completed():
    started = datetime.now(timezone.utc)
    completed = started + timedelta(milliseconds=250)
    r = ToolResult(
        success=True,
        tool_call_id="abc",
        started_at=started,
        completed_at=completed,
    )
    # Allow ±1ms slack for rounding.
    assert 249 <= (r.duration_ms or 0) <= 251


def test_duration_ms_none_if_not_completed():
    r = ToolResult(success=True, tool_call_id="abc")
    assert r.duration_ms is None


# -------- success_result FACTORY -----------------------------------------------------------
def test_success_result_basic():
    r = ToolResult.success_result(tool_call_id="abc", result={"foo": "bar"})
    assert r.success is True
    assert r.tool_call_id == "abc"
    assert r.result == {"foo": "bar"}
    assert r.error is None
    assert r.failure_reason is None
    assert r.completed_at is not None  # factory sets it


def test_success_result_with_metadata():
    r = ToolResult.success_result(
        tool_call_id="abc", result=42, metadata={"source": "test"}
    )
    assert r.metadata == {"source": "test"}


def test_success_result_metadata_defaults_to_empty():
    r = ToolResult.success_result(tool_call_id="abc", result=None)
    assert r.metadata == {}


# -------- tool_failure FACTORY -----------------------------------------------------------
def test_tool_failure_default_reason():
    r = ToolResult.tool_failure(tool_call_id="abc", error="boom")
    assert r.success is False
    assert r.error == "boom"
    assert r.failure_reason == FailureReason.EXECUTION_ERROR
    assert r.result is None


def test_tool_failure_custom_reason():
    r = ToolResult.tool_failure(
        tool_call_id="abc",
        error="oops",
        reason=FailureReason.TIMEOUT,
    )
    assert r.failure_reason == FailureReason.TIMEOUT


# -------- CONVENIENCE FACTORIES -----------------------------------------------------------
def test_cancelled_before_start():
    r = ToolResult.cancelled_before_start("abc")
    assert r.success is False
    assert r.failure_reason == FailureReason.CANCELLED_BEFORE_START
    assert "cancelled before" in (r.error or "").lower()


def test_cancelled_during_execution():
    r = ToolResult.cancelled_during_execution("abc")
    assert r.success is False
    assert r.failure_reason == FailureReason.CANCELLED_DURING_EXECUTION
    assert "cancelled during" in (r.error or "").lower()


def test_invalid_parameters():
    r = ToolResult.invalid_parameters("abc", "missing 'query'")
    assert r.success is False
    assert r.failure_reason == FailureReason.INVALID_PARAMETERS
    assert "missing 'query'" in (r.error or "")


def test_execution_error_with_message():
    r = ToolResult.execution_error("abc", "db connection lost")
    assert r.success is False
    assert r.failure_reason == FailureReason.EXECUTION_ERROR
    assert r.error == "db connection lost"


def test_execution_error_default_message():
    r = ToolResult.execution_error("abc")
    assert r.success is False
    assert r.failure_reason == FailureReason.EXECUTION_ERROR
    assert r.error  # has some default text


def test_timeout_factory():
    r = ToolResult.timeout("abc", timeout_seconds=30)
    assert r.success is False
    assert r.failure_reason == FailureReason.TIMEOUT
    assert "30" in (r.error or "")


# -------- SERIALIZATION -----------------------------------------------------------
def test_round_trip_serialization():
    r = ToolResult.success_result(tool_call_id="abc", result=[1, 2, 3])
    dumped = r.model_dump()
    restored = ToolResult.model_validate(dumped)
    assert restored == r