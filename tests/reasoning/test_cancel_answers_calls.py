"""A cancelled run still answers every tool call, so the next turn works."""

import asyncio
from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage
from max_ai.core.termination import CancellationToken
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

RAN: list[str] = []


async def slow() -> str:
    """Slow work."""
    RAN.append("slow")
    await asyncio.sleep(5)
    return "slow done"


async def quick() -> str:
    """Quick read."""
    RAN.append("quick")
    return "quick done"


async def later() -> str:
    """Work that comes after the slow call."""
    RAN.append("later")
    return "later done"


class StrictClient:
    """Rejects a request with an unanswered tool call, like OpenAI's 400.
    The first reply makes ``calls``; every other one answers."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, *calls: str):
        self.calls = list(calls)
        self.replies = 0

    async def run(self, ctx, prompts, tools=None, **kwargs):
        answered = {m.tool_call_id for m in ctx.messages if isinstance(m, ToolMessage)}
        for message in ctx.messages:
            for call in getattr(message, "tool_calls", None) or []:
                if call.id not in answered:
                    raise RuntimeError(f"400: tool call {call.id} has no response")
        self.replies += 1
        calls = [ToolCall(id=name, tool_name=name, parameters={}) for name in self.calls]
        if self.replies > 1:
            calls = []
        message = AssistantMessage(source="fake", content="" if calls else "answer", tool_calls=calls)
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="tool_calls" if calls else "stop")


def make_agent(tmp_path, client, *, read_only=False):
    tools = [FunctionAsTool(f, approval_mode=ToolApprovalMode.AUTO_APPROVED, read_only=read_only)
             for f in (slow, quick, later)]
    agent = Agent(name="t", description="d", instructions="i", client=client, toolset=tools,
                  workspace=LocalWorkspace(root=tmp_path))
    agent.reasoning.max_connection_retries = 0
    return agent


async def cancel_then_ask_again(agent, ctx):
    """Cancel the first run while ``slow`` runs, then send another message."""
    RAN.clear()
    token = CancellationToken()
    task = asyncio.create_task(agent.run("work", run_context=ctx, cancellation_token=token))
    while "slow" not in RAN:
        await asyncio.sleep(0.01)
    token.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return await agent.run("and now?", run_context=ctx)


def answers(ctx) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in ctx.messages if isinstance(m, ToolMessage)}


async def test_a_call_cancelled_while_running_is_answered(tmp_path):
    ctx = RunContext(user_id="u", session_id="s")
    async with make_agent(tmp_path, StrictClient("slow")) as agent:
        response = await cancel_then_ask_again(agent, ctx)

    assert response.final_text == "answer"
    assert not answers(ctx)["slow"].success


async def test_parallel_reads_keep_the_results_that_finished(tmp_path):
    ctx = RunContext(user_id="u", session_id="s")
    async with make_agent(tmp_path, StrictClient("quick", "slow"), read_only=True) as agent:
        response = await cancel_then_ask_again(agent, ctx)

    assert response.final_text == "answer"
    got = answers(ctx)
    assert got["quick"].success and "quick done" in got["quick"].text()
    assert not got["slow"].success


async def test_a_call_that_never_started_does_not_run_on_the_next_turn(tmp_path):
    ctx = RunContext(user_id="u", session_id="s")
    async with make_agent(tmp_path, StrictClient("slow", "later")) as agent:
        response = await cancel_then_ask_again(agent, ctx)

    assert response.final_text == "answer"
    assert "later" not in RAN
    assert not answers(ctx)["later"].success
