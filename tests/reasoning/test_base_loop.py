"""
Unit tests for BaseLoopState and the streaming helpers in BaseReasoning.

These tests don't run a full reasoning loop — they isolate:
  - State accumulation (record_usage, record_completion, retries).
  - Tool-call chunk merging across streamed fragments.
  - Tool-call assembly from accumulated fragments (drop on missing
    name / malformed JSON arguments).

Loop-level behavior (iterations, tool dispatch, approval pauses) lives
in test_react_loop.py.
"""

from __future__ import annotations

import logging
import pytest
import typing as t 

from max_ai.base.reasoning import BaseReasoning, BaseLoopState
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.types.completions import ChatCompletionResult, Usage


# -------- BASE LOOP STATE -----------------------------------------------------------
def test_base_loop_state_defaults():
    state = BaseLoopState()
    assert state.iteration == 0
    assert state.finish_reason == "unknown"
    assert state.llm_calls == 0
    assert state.tool_calls == 0
    assert state.attempts_to_call_api == 0
    assert state.tokens_input == 0
    assert state.tokens_output == 0
    assert state.last_result is None
    assert state.retries == 0


def test_record_usage_accumulates():
    state = BaseLoopState()
    state.record_usage(
        Usage(
            llm_calls=1,
            attempts_to_call_api=1,
            tokens_input=100,
            tokens_output=50,
            tokens_cached=10,
        )
    )
    state.record_usage(
        Usage(
            llm_calls=1,
            attempts_to_call_api=2,
            tokens_input=200,
            tokens_output=75,
            tokens_cached=5,
        )
    )
    assert state.llm_calls == 2
    assert state.attempts_to_call_api == 3
    assert state.tokens_input == 300
    assert state.tokens_output == 125
    assert state.tokens_cached == 15


def test_retries_property():
    state = BaseLoopState(llm_calls=2, attempts_to_call_api=5)
    assert state.retries == 3


def test_retries_never_negative():
    """If attempts < calls (shouldn't happen, but defensive), retries clamps to 0."""
    state = BaseLoopState(llm_calls=5, attempts_to_call_api=2)
    assert state.retries == 0


def test_record_completion_stores_result_and_accumulates_usage():
    state = BaseLoopState()
    msg = AssistantMessage(source="agent", content="hello")
    result = ChatCompletionResult(
        message=msg,
        usage=Usage(llm_calls=1, attempts_to_call_api=1, tokens_input=10),
        model="test-model",
        finish_reason="stop",
    )
    state.record_completion(result)
    assert state.last_result is result
    assert state.llm_calls == 1
    assert state.tokens_input == 10


# -------- TOOL CALL CHUNK MERGING -----------------------------------------------------------
def test_merge_single_complete_chunk():
    """Ollama-style: one chunk with the whole tool call."""
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        },
    )
    assert "call_1" in accumulated
    assert accumulated["call_1"]["function"]["name"] == "get_weather"
    assert accumulated["call_1"]["function"]["arguments"] == '{"city": "NYC"}'


def test_merge_argument_deltas():
    """OpenAI-style: arguments arrive in fragments, must concatenate."""
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"ci'}},
    )
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {"id": "call_1", "function": {"arguments": 'ty": "NYC"}'}},
    )
    assert accumulated["call_1"]["function"]["name"] == "get_weather"
    assert accumulated["call_1"]["function"]["arguments"] == '{"city": "NYC"}'


def test_merge_name_arrives_late():
    """Some providers send id first, name later."""
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated, {"id": "call_1", "function": {"arguments": ""}}
    )
    BaseReasoning._merge_tool_call_chunk(
        accumulated, {"id": "call_1", "function": {"name": "ping"}}
    )
    assert accumulated["call_1"]["function"]["name"] == "ping"


def test_merge_multiple_calls():
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {"id": "call_1", "function": {"name": "a", "arguments": "{}"}},
    )
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {"id": "call_2", "function": {"name": "b", "arguments": "{}"}},
    )
    assert set(accumulated) == {"call_1", "call_2"}


def test_merge_drops_chunk_without_id():
    """Fragments without an id can't be associated — silently dropped."""
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated, {"function": {"name": "orphan"}}
    )
    assert accumulated == {}


def test_merge_does_not_overwrite_existing_name():
    """If the name was already set by an earlier fragment, don't clobber it."""
    accumulated: dict[str, dict[str, t.Any]] = {}
    BaseReasoning._merge_tool_call_chunk(
        accumulated,
        {"id": "call_1", "function": {"name": "first", "arguments": ""}},
    )
    BaseReasoning._merge_tool_call_chunk(
        accumulated, {"id": "call_1", "function": {"name": "second"}}
    )
    assert accumulated["call_1"]["function"]["name"] == "first"


# -------- TOOL CALL BUILD FROM CHUNKS -----------------------------------------------------------
@pytest.fixture
def silent_log():
    """Scoped logger that won't pollute output during drop-warning tests."""
    from max_ai.loggers import ScopedLogger

    return ScopedLogger(logging.getLogger("test.silent"), scope="test")


def test_build_tool_calls_happy_path(silent_log):
    accumulated = {
        "call_1": {
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        },
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert len(calls) == 1
    tc = calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_1"
    assert tc.tool_name == "get_weather"
    assert tc.parameters == {"city": "NYC"}


def test_build_tool_calls_empty_arguments(silent_log):
    """Empty arguments string → empty dict (treated as no params)."""
    accumulated = {
        "call_1": {
            "id": "call_1",
            "function": {"name": "ping", "arguments": ""},
        },
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert len(calls) == 1
    assert calls[0].parameters == {}


def test_build_tool_calls_drops_missing_name(silent_log):
    accumulated = {
        "call_1": {
            "id": "call_1",
            "function": {"name": None, "arguments": '{"x": 1}'},
        },
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert calls == []


def test_build_tool_calls_drops_malformed_json(silent_log):
    accumulated = {
        "call_1": {
            "id": "call_1",
            "function": {"name": "do_thing", "arguments": "not-valid-json{"},
        },
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert calls == []


def test_build_tool_calls_partial_drop(silent_log):
    """Good calls survive even when bad ones are dropped."""
    accumulated = {
        "call_good": {
            "id": "call_good",
            "function": {"name": "ok", "arguments": "{}"},
        },
        "call_bad": {
            "id": "call_bad",
            "function": {"name": "broken", "arguments": "{not json"},
        },
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert len(calls) == 1
    assert calls[0].tool_name == "ok"


def test_build_tool_calls_multiple(silent_log):
    accumulated = {
        "c1": {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
        "c2": {"id": "c2", "function": {"name": "b", "arguments": '{"y": 2}'}},
    }
    calls = BaseReasoning._build_tool_calls_from_chunks(accumulated, silent_log)
    assert {c.tool_name for c in calls} == {"a", "b"}