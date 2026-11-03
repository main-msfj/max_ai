"""At the iteration limit the turn still ends with a reply to the user."""

from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.capabilities.reasoning.react import ReactLoop
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.harness import messages as harness
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


def step() -> str:
    """One step of a long task."""
    return "ok"


class EndlessClient:
    """Calls a tool whenever it may; without tools it answers ``final``."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, final: str):
        self.final = final
        self.tools = []

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.tools.append(list(tools or []))
        if tools:
            message = AssistantMessage(source="fake", content="", tool_calls=[
                ToolCall(tool_name="step", parameters={})])
            return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                        finish_reason="tool_calls")
        # A provider may send a call even with no tools offered.
        message = AssistantMessage(source="fake", content=self.final, tool_calls=[
            ToolCall(tool_name="step", parameters={})])
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="stop")


async def run_to_limit(tmp_path, client):
    agent = Agent(name="t", description="d", instructions="i", client=client,
                  reasoning=ReactLoop(max_loop_iterations=2),
                  toolset=[FunctionAsTool(step, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
                  workspace=LocalWorkspace(root=tmp_path))
    ctx = RunContext(user_id="u", session_id="s")
    async with agent:
        response = await agent.run("do the long task", run_context=ctx)
    return response, ctx


async def test_the_limit_ends_with_an_answer_made_without_tools(tmp_path):
    client = EndlessClient(final="Did 2 steps; the rest is missing.")
    response, ctx = await run_to_limit(tmp_path, client)

    assert response.finish_reason == "max_iterations"
    assert [bool(tools) for tools in client.tools] == [True, True, False]
    last = ctx.messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.text() == "Did 2 steps; the rest is missing."
    assert last.tool_calls == []


async def test_no_answer_leaves_the_note_for_the_next_turn(tmp_path):
    response, ctx = await run_to_limit(tmp_path, EndlessClient(final=""))

    assert response.finish_reason == "max_iterations"
    last = ctx.messages[-1]
    assert isinstance(last, UserMessage)
    assert last.text() == f"[harness] {harness.MAX_ITERATIONS_REACHED}"
