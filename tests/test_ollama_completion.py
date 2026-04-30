"""
Integration test for OllamaChatCompletionClient.

Matrix:
    - qwen3:4b-thinking-2507-q4_K_M  with think=False and think=True
    - phi4-mini:3.8b                 with think=True
    - each combination tested in both complete and stream mode

Requires Ollama running locally and both models pulled:

    docker exec -it <ollama-container> ollama pull qwen3:4b-thinking-2507-q4_K_M
    docker exec -it <ollama-container> ollama pull phi4-mini:3.8b

Run:
    python -m tests.integration.test_ollama_client
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest

from max_ai.base.tool_executor import ToolExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.clients.ollama.client import OllamaChatCompletionClient
from max_ai.core.messages import UserMessage
from max_ai.core.models import ModelConfig
from max_ai.middleware.chain import MiddlewareChain
from max_ai.reasoning.react import ReActLoop, ReActLoopState
from max_ai.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.tools import ToolApprovalMode
from max_ai.manager.stacks import build_default_stack


# -------- TEST CONFIGURATION -----------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """A single (model, think, prompt, expect_thinking) combination."""

    label: str
    model: str
    think: bool
    prompt: str
    expect_thinking: bool

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        label="phi4-mini no-think",
        model="phi4-mini:3.8b",
        think=False,
        prompt="What is 12 * 9? Think briefly, then give just the number.",
        expect_thinking=False,
    ),
    Scenario(
        label="gemma4:e2b no-think",
        model="gemma4:e2b-it-q4_K_M",
        think=False,
        prompt="Say 'hello world' in exactly two words.",
        expect_thinking=False,
    ),
    Scenario(
        label="gemma4:e2b think",
        model="gemma4:e2b-it-q4_K_M",
        think=True,
        prompt="What is 1234 * 5678? Show your reasoning step by step.",
        # gemma4:e2b appears to not emit native thinking even with think=True.
        # Marking as False so the test reflects observed behaviour.
        expect_thinking=True,
    ),
    Scenario(
        label="qwen3 no-think",
        model="qwen3:4b-thinking-2507-q4_K_M",
        think=False,
        prompt="Say 'hello world' in exactly two words.",
        # qwen3-thinking is a reasoning-specialised model that emits
        # thinking regardless of the think flag. Ollama's SDK still
        # separates it into the thinking field. We accept this.
        expect_thinking=False,
    ),
    Scenario(
        label="qwen3 think",
        model="qwen3:4b-thinking-2507-q4_K_M",
        think=True,
        prompt="Say 'hello world' in exactly two words.",
        # qwen3-thinking is a reasoning-specialised model that emits
        # thinking regardless of the think flag. Ollama's SDK still
        # separates it into the thinking field. We accept this.
        expect_thinking=True,
    )
)

# -------- HELPERS -----------------------------------------------------------
def make_prompt_ctx() -> PromptCtx:
    """Minimal PromptCtx with a tiny system prompt under AgentPolicy."""
    stack = build_default_stack()
    rendered_layers: dict[type, str] = {}
    for layer in stack:
        layer_type = type(layer)
        if layer_type.__name__ == "AgentPolicyLayer":
            rendered_layers[layer_type] = (
                "You are a helpful assistant. Answer concisely."
            )
        else:
            rendered_layers[layer_type] = ""
    return PromptCtx(
        stack=stack,
        variables={},
        rendered_layers=rendered_layers,
    )


def make_run_ctx(user_text: str) -> RunContext:
    return RunContext(
        session_id="integration-test",
        messages=[UserMessage(source="user", content=user_text)]
    )


def make_client(scenario: Scenario) -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=scenario.model,
        host=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        config=ModelConfig(
            thinking_tag="think",
            thinking_position="start",
            supports_thinking=scenario.think
        ),
        think=scenario.think if scenario.think else None,
    )


# -------- ASSERTIONS -----------------------------------------------------------
def check_complete(scenario: Scenario, content: str, thinking: str | None) -> list[str]:
    """Return a list of failure descriptions (empty list = pass)."""
    failures: list[str] = []

    if not content or not content.strip():
        failures.append("content is empty")

    if scenario.expect_thinking and not thinking:
        failures.append("expected thinking but got None / empty")
    if not scenario.expect_thinking and thinking:
        failures.append(f"did not expect thinking but got: {thinking[:60]!r}")

    # Thinking should never leak into content as raw tags.
    for tag in ("<think>", "</think>"):
        if tag in content:
            failures.append(f"raw tag {tag!r} leaked into content")

    return failures


# -------- RUNNERS -----------------------------------------------------------
async def run_complete(scenario: Scenario) -> bool:
    print(f"\n--- COMPLETE: {scenario.label} ---")
    client = make_client(scenario)
    ctx = make_run_ctx(scenario.prompt)
    prompts = make_prompt_ctx()

    result = await client.run(ctx, prompts, stream=False)

    content = result.message.content if isinstance(result.message.content, str) else result.message.text()
    thinking = result.message.thinking

    print(f"  content:  {content!r}")
    print(f"  thinking: {(thinking[:120] + '...') if thinking and len(thinking) > 120 else thinking!r}")
    print(f"  usage:    in={result.usage.tokens_input} out={result.usage.tokens_output} dur={result.usage.duration_ms}ms")

    failures = check_complete(scenario, content, thinking)
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return False
    print("  ✓ pass")
    return True


async def run_stream(scenario: Scenario) -> bool:
    print(f"\n--- STREAM:   {scenario.label} ---")
    client = make_client(scenario)
    ctx = make_run_ctx(scenario.prompt)
    prompts = make_prompt_ctx()

    thinking_pieces: list[str] = []
    content_pieces: list[str] = []
    final_chunk = None
    saw_chunks = 0

    async for chunk in await client.run(ctx, prompts, stream=True):
        saw_chunks += 1
        if chunk.is_complete:
            final_chunk = chunk
            continue
        if chunk.thinking:
            thinking_pieces.append(chunk.thinking)
        if chunk.content:
            content_pieces.append(chunk.content)

    content = "".join(content_pieces)
    thinking = "".join(thinking_pieces) if thinking_pieces else None

    print(f"  chunks:   {saw_chunks}")
    print(f"  content:  {content!r}")
    print(f"  thinking: {(thinking[:120] + '...') if thinking and len(thinking) > 120 else thinking!r}")
    if final_chunk and final_chunk.usage:
        u = final_chunk.usage
        print(f"  usage:    in={u.tokens_input} out={u.tokens_output} dur={u.duration_ms}ms")

    failures: list[str] = []
    if final_chunk is None:
        failures.append("no final chunk with is_complete=True")
    elif final_chunk.usage is None:
        failures.append("final chunk missing usage")
    failures.extend(check_complete(scenario, content, thinking))

    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return False
    print("  ✓ pass")
    return True


# -------- MAIN -----------------------------------------------------------
async def main() -> None:
    results: list[tuple[str, bool]] = []

    for scenario in SCENARIOS:
        ok = await run_complete(scenario)
        results.append((f"{scenario.label} / complete", ok))

    for scenario in SCENARIOS:
        ok = await run_stream(scenario)
        results.append((f"{scenario.label} / stream", ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for label, ok in results:
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {label}")
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())


class LuckyNumberTool(CoreTool):
    """Tiny real tool for an opt-in ReAct smoke test."""

    def __init__(self) -> None:
        super().__init__(
            name="lookup_lucky_number",
            description="Return the user's lucky number for smoke tests.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=30,
        )
        self.execute_called = False

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        self.execute_called = True
        return ToolResult.success_result(tool_call_id=tool_request.id, result="7")


def make_react_prompt_ctx() -> PromptCtx:
    stack = build_default_stack()
    rendered_layers: dict[type, str] = {}
    for layer in stack:
        layer_type = type(layer)
        if layer_type.__name__ == "AgentPolicyLayer":
            rendered_layers[layer_type] = (
                "You are a careful assistant. "
                "When the user asks for their lucky number and the tool "
                "'lookup_lucky_number' is available, call that tool before answering. "
                "After the tool returns, answer with the tool result plainly."
            )
        else:
            rendered_layers[layer_type] = ""
    return PromptCtx(stack=stack, variables={}, rendered_layers=rendered_layers)


@pytest.mark.asyncio
async def test_phi4_mini_react_cycle_smoke():
    if os.getenv("MAX_AI_RUN_OLLAMA_REACT") != "1":
        pytest.skip("Set MAX_AI_RUN_OLLAMA_REACT=1 to run the live Ollama ReAct smoke test.")

    client = OllamaChatCompletionClient(
        model="phi4-mini:3.8b",
        host=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        config=ModelConfig(
            supports_function_calling=True,
            supports_thinking=False,
            thinking_tag="think",
            thinking_position="start",
        ),
        think=False,
    )
    tool = LuckyNumberTool()
    loop = ReActLoop(
        name="ollama-react-smoke",
        client=client,
        tool_executor=ToolExecutor(tools=[tool], agent_name="ollama-react-smoke"),
        middleware_chain=MiddlewareChain(),
        max_loop_iterations=4,
    )
    ctx = RunContext(
        session_id="ollama-react-smoke",
        messages=[
            UserMessage(
                source="user",
                content=(
                    "What is my lucky number? Use the lookup_lucky_number tool first, "
                    "then answer with only the final number."
                ),
            )
        ],
    )
    state = ReActLoopState()

    events = [event async for event in loop.execute_reasoning_loop(ctx, make_react_prompt_ctx(), state)]

    assert events
    assert tool.execute_called is True
    assert state.tool_calls >= 1
    assert len(ctx.messages) >= 4
    assert ctx.messages[-1].role == "assistant"
    assert "7" in ctx.messages[-1].text()
