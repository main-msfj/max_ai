from __future__ import annotations

import os
import typing as t

import pytest

from max_ai.base.tool_executor import ToolExecutor
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.clients.ollama.client import OllamaChatCompletionClient
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.core.models import ModelConfig
from max_ai.reasoning.react import ReActLoop, ReActLoopState
from max_ai.termination import CancellationToken
from max_ai.types.middleware import MiddlewareCtx
from max_ai.types.run_context import RunContext
from max_ai.types.stacks import PromptCtx
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode
from max_ai.validators.stacks import build_default_stack

MiddlewareCtx.model_rebuild(_types_namespace={"RunContext": RunContext})


class LuckyNumberTool(CoreTool):
    def __init__(self) -> None:
        super().__init__(
            name="lookup_lucky_number",
            description="Return the user's lucky number.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            timeout_seconds=30,
        )
        self.execute_called = False

    @property
    def parameters(self) -> dict[str, t.Any]:
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


def make_prompts() -> PromptCtx:
    stack = build_default_stack()
    rendered_layers: dict[type, str] = {}
    for layer in stack:
        if type(layer).__name__ == "AgentPolicyLayer":
            rendered_layers[type(layer)] = (
                "You are a careful assistant. "
                "If the user asks for a lucky number and the tool "
                "'lookup_lucky_number' exists, call it first. "
                "Then answer using only the tool result."
            )
        else:
            rendered_layers[type(layer)] = ""
    return PromptCtx(stack=stack, variables={}, rendered_layers=rendered_layers)


def make_ctx() -> RunContext:
    return RunContext(
        session_id="phi4-react-cycle",
        messages=[
            UserMessage(
                source="user",
                content="What is my lucky number? Use the tool first.",
            )
        ],
    )


@pytest.mark.asyncio
async def test_phi4_mini_react_cycle():
    if os.getenv("MAX_AI_RUN_OLLAMA_REACT") != "1":
        pytest.skip("Set MAX_AI_RUN_OLLAMA_REACT=1 to run the live phi4 smoke test.")

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
    executor = ToolExecutor(tools=[tool], agent_name="phi4-react")
    loop = ReActLoop(
        name="phi4-react",
        client=client,
        tool_executor=executor,
        middleware_chain=executor.mw_chain,
        max_loop_iterations=4,
    )
    ctx = make_ctx()
    state = ReActLoopState()

    events = [
        event
        async for event in loop.execute_reasoning_loop(
            ctx=ctx,
            prompts=make_prompts(),
            loop_state=state,
        )
    ]

    assert tool.execute_called is True
    assert state.tool_calls >= 1
    assert state.llm_calls >= 1
    assert ctx.tool_state.pending_approvals == []
    assert isinstance(ctx.messages[-1], AssistantMessage)
    assert "7" in ctx.messages[-1].text()
    assert any(getattr(e, "EVENT_TYPE", "") == "tool_call" for e in events)
