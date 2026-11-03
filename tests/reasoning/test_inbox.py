"""Messages the user sends mid-turn reach the next model call."""

from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

CTX = RunContext(user_id="u", session_id="s")


async def lookup() -> str:
    """Look something up."""
    CTX.inbox.append("also in euros")
    return "42 dollars"


class ScriptedClient:
    """Plays ``replies`` in order and records the user messages it saw."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.seen.append([m.text() for m in ctx.messages if isinstance(m, UserMessage)])
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply()
        calls = [ToolCall(id="c1", tool_name=reply, parameters={})] if reply == "lookup" else []
        message = AssistantMessage(source="fake", content="" if calls else reply, tool_calls=calls)
        return ChatCompletionResult(message=message, usage=Usage(), model=self.model,
                                    finish_reason="tool_calls" if calls else "stop")


def make_agent(tmp_path, client):
    tool = FunctionAsTool(lookup, approval_mode=ToolApprovalMode.AUTO_APPROVED)
    return Agent(name="t", description="d", instructions="i", client=client,
                 toolset=[tool], workspace=LocalWorkspace(root=tmp_path))


async def test_a_message_sent_during_a_tool_call_joins_the_next_call(tmp_path):
    CTX.messages.clear()
    client = ScriptedClient("lookup", "42 dollars, 39 euros")
    async with make_agent(tmp_path, client) as agent:
        response = await agent.run("price?", run_context=CTX)

    assert client.seen == [["price?"], ["price?", "also in euros"]]
    assert response.final_text == "42 dollars, 39 euros"
    assert CTX.inbox == []


async def test_a_message_sent_during_the_answer_gets_its_own_answer(tmp_path):
    ctx = RunContext(user_id="u", session_id="s")

    def answer_while_user_writes():
        ctx.inbox.append("and tomorrow?")
        return "sunny"

    client = ScriptedClient(answer_while_user_writes, "rain tomorrow")
    async with make_agent(tmp_path, client) as agent:
        response = await agent.run("weather?", run_context=ctx)

    assert client.seen == [["weather?"], ["weather?", "and tomorrow?"]]
    assert [m.text() for m in ctx.messages if isinstance(m, AssistantMessage)] == [
        "sunny", "rain tomorrow",
    ]
    assert response.final_text == "rain tomorrow"
