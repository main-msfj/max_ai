"""End-to-end integration test for the full approval flow.

Validates the complete cycle: pause → persist → load → approve → resume.

Requires:
  - Ollama reachable at $OLLAMA_HOST (default http://ollama:11434).
  - qwen3:4b-thinking-2507-q4_K_M (or another tool-capable model) pulled.

Run with: pytest -m integration tests/integration/test_ollama_approval_flow.py
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from max_ai.base.agent import Agent
from max_ai.clients.ollama.client import OllamaChatCompletionClient
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.persistence.filesystem import FileSystemRunContextStore
from max_ai.core.models import ModelConfig
from max_ai.core.messages import AssistantMessage, ToolMessage, UserMessage
from max_ai.types.tools import ToolApprovalMode


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


# -------- HEALTH CHECK -----------------------------------------------------------
def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason=f"Ollama not reachable at {OLLAMA_HOST}",
    ),
]


# -------- FIXTURES -----------------------------------------------------------
@pytest.fixture
def client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=MODEL,
        host=OLLAMA_HOST,
        config=ModelConfig(
            supports_function_calling=True,
            supports_thinking=False,
        ),
        think=False,
    )


def _make_weather_tool_ask_approval() -> FunctionAsTool:
    """A weather tool that requires approval before running."""

    def get_weather(city: str) -> str:
        """Return the current weather in the given city."""
        return f"Sunny, 22°C in {city}"

    return FunctionAsTool(
        func=get_weather,
        name="get_weather",
        description="Get the current weather for a city.",
        approval_mode=ToolApprovalMode.ASK_APPROVED,
    )


def _build_agent(client: OllamaChatCompletionClient, tool: FunctionAsTool) -> Agent:
    return Agent(
        name="weather_agent",
        description="Reports weather for cities, with approval.",
        instructions=(
            "You have a get_weather tool. When asked about weather, "
            "call the tool and report the result to the user."
        ),
        client=client,
        toolset=[tool],
    )


# -------- TEST -----------------------------------------------------------
@pytest.mark.asyncio
async def test_full_approval_cycle_approved(client, tmp_path):
    """Complete cycle with the user approving the tool call."""
    store = FileSystemRunContextStore(base_path=tmp_path)
    tool = _make_weather_tool_ask_approval()
    agent = _build_agent(client, tool)

    # ── Turn 1: agent runs, pauses on approval ───────────────────────────
    response = await agent.run(task="What is the weather in Tokyo?")

    assert response.finish_reason == "approval_needed"
    assert response.needs_approval
    assert len(response.pending_approvals) >= 1

    pending = response.pending_approvals[0]
    assert pending.tool_name == "get_weather"
    run_id = response.context.run_id

    # ── Persist ──────────────────────────────────────────────────────────
    await store.save(run_id, response.context)
    assert run_id in await store.list()

    # ── Simulate process death + restart ─────────────────────────────────
    del response  # caller doesn't have the in-memory state anymore

    ctx = await store.load(run_id)
    assert ctx is not None
    assert ctx.tool_state.waiting_for_approval

    # ── User approves ────────────────────────────────────────────────────
    for record in ctx.tool_state.pending_approvals:
        ctx.tool_state.apply_approval(
            record.id, approved=True, reason="approved in test"
        )

    assert not ctx.tool_state.waiting_for_approval
    assert len(ctx.tool_state.actionable_calls) >= 1

    # ── Resume ───────────────────────────────────────────────────────────
    final = await agent.resume(run_context=ctx)

    assert final.finish_reason == "stop"
    assert final.usage.tool_calls >= 1

    # Transcript sanity: at least one tool message + a final assistant reply.
    has_tool_msg = any(isinstance(m, ToolMessage) for m in final.messages)
    assert has_tool_msg, "no ToolMessage in resumed transcript"

    final_msg = final.final_message
    assert final_msg is not None
    assert not final_msg.tool_calls
    text = final_msg.text().lower()
    assert "tokyo" in text or "22" in text or "sunny" in text

    # All records should be consumed by now.
    records = list(final.context.tool_state.records.values())
    assert all(r.is_consumed for r in records)

    # ── Cleanup ──────────────────────────────────────────────────────────
    await store.delete(run_id)
    assert run_id not in await store.list()


@pytest.mark.asyncio
async def test_full_approval_cycle_rejected(client, tmp_path):
    """Complete cycle with the user rejecting the tool call."""
    store = FileSystemRunContextStore(base_path=tmp_path)
    tool = _make_weather_tool_ask_approval()
    agent = _build_agent(client, tool)

    # Turn 1
    response = await agent.run(task="What is the weather in Tokyo?")
    assert response.needs_approval
    run_id = response.context.run_id

    # Persist + reload
    await store.save(run_id, response.context)
    ctx = await store.load(run_id)
    assert ctx is not None

    # Reject
    for record in ctx.tool_state.pending_approvals:
        ctx.tool_state.apply_approval(
            record.id, approved=False, reason="rejected in test"
        )

    # Resume — the LLM should produce a closing reply without the tool result.
    final = await agent.resume(run_context=ctx)

    print(f"\n=== finish_reason: {final.finish_reason}")
    print(f"=== iterations: {final.usage.llm_calls}")
    for r in final.context.tool_state.records.values():
        print(f"=== record {r.id[:8]} | tool={r.tool_name} | status={r.status} | success={r.result.success if r.result else 'no result'}")
    print(f"=== final_text: {final.final_message.text() if final.final_message else 'NONE'}")

    assert final.finish_reason == "stop"
    # Tool was never executed.
    records = list(final.context.tool_state.records.values())
    assert all(r.is_consumed for r in records)
    assert all(not r.result.success for r in records if r.result is not None)

    # The agent should have produced a final assistant message.
    assert final.final_message is not None
    assert not final.final_message.tool_calls
