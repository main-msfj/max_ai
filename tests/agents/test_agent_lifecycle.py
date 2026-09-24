"""The Agent's run lifecycle: answer, stream, fail, cancel, pause for
approvals and questions, and resume with the user's decisions."""

from __future__ import annotations

import asyncio

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.event_type import ErrorEvent, ModelCallEvent, ModelResponseEvent
from max_ai.core.messages import AssistantMessage, ToolCall, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.termination import CancellationToken
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

PAID: list[int] = []


def pay(amount: int) -> dict:
    """Pay an amount. Has a real side effect."""
    PAID.append(amount)
    return {"paid": amount}


class ScriptedLLM:
    model = "scripted"
    config = ModelConfig()
    generation_options = {"max_tokens": 200}

    def __init__(self, *replies: AssistantMessage, fail: bool = False, delay: float = 0):
        self.replies, self.fail, self.delay = list(replies), fail, delay

    async def run(self, **kwargs):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider down")
        message = self.replies.pop(0)
        if kwargs.get("stream"):
            async def chunks():
                for word in message.content.split(" "):
                    yield ChatCompletionChunk(content=word + " ", is_complete=False)
                yield ChatCompletionChunk(content="", is_complete=True, usage=Usage())
            return chunks()
        return ChatCompletionResult(message=message, usage=Usage(), model="scripted", finish_reason="stop")


def say(text: str) -> AssistantMessage:
    return AssistantMessage(source="llm", content=text)


def pay_calls(*amounts: int) -> AssistantMessage:
    return AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id=f"pay-{amount}", tool_name="pay", parameters={"amount": amount}) for amount in amounts])


@pytest.fixture(autouse=True)
def reset():
    PAID.clear()


def agent(tmp_path, llm) -> Agent:
    tool = FunctionAsTool(pay, approval_mode=ToolApprovalMode.ASK_APPROVED)
    a = Agent(name="a", description="d", instructions="i", client=llm, toolset=[tool],
              workspace=LocalWorkspace(root=tmp_path))
    a.reasoning.max_connection_retries = 0
    return a


# -------- RUN -----------------------------------------------------------
async def test_run_answers_and_streams_events_in_order(tmp_path):
    async with agent(tmp_path, ScriptedLLM(say("hola"), say("otra"))) as a:
        response = await a.run("hola", run_context=RunContext(user_id="u"))
        assert (response.finish_reason, response.final_text) == ("stop", "hola")

        items = [item async for item in a.run_stream_events("¿y?", run_context=RunContext(user_id="u"))]
    assert isinstance(items[-1], AgentResponse)
    kinds = [type(item) for item in items]
    assert kinds.index(ModelCallEvent) < kinds.index(ModelResponseEvent)


async def test_run_stream_yields_only_text(tmp_path):
    async with agent(tmp_path, ScriptedLLM(say("solo texto"))) as a:
        chunks = [chunk async for chunk in a.run_stream("hola", run_context=RunContext(user_id="u"))]
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks).strip() == "solo texto"


async def test_a_provider_error_is_emitted_then_raised(tmp_path):
    events = []
    async with agent(tmp_path, ScriptedLLM(fail=True)) as a:
        with pytest.raises(RuntimeError, match="provider down"):
            async for item in a.run_stream_events("hola", run_context=RunContext(user_id="u")):
                events.append(item)
    assert isinstance(events[-1], ErrorEvent)


async def test_cancellation_stops_the_run(tmp_path):
    token = CancellationToken()
    async with agent(tmp_path, ScriptedLLM(say("nunca"), delay=5)) as a:
        task = asyncio.create_task(a.run("hola", run_context=RunContext(user_id="u"), cancellation_token=token))
        await asyncio.sleep(0.1)
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# -------- APPROVALS -----------------------------------------------------------
async def test_a_tool_that_needs_approval_pauses_and_resumes_when_approved(tmp_path):
    ctx = RunContext(user_id="u")
    async with agent(tmp_path, ScriptedLLM(pay_calls(10), say("pagado"))) as a:
        paused = await a.run("paga 10", run_context=ctx)
        assert paused.finish_reason == "approval_needed" and PAID == []
        assert [r.tool_name for r in paused.pending_approvals] == ["pay"]

        with pytest.raises(ValueError, match="Resolve pending tool calls"):
            await a.run("otra cosa", run_context=ctx)

        paused.pending_approvals[0].approve()
        done = await a.run(run_context=ctx)  # no new task: resume
    assert (done.finish_reason, done.final_text, PAID) == ("stop", "pagado", [10])


async def test_only_approved_calls_run(tmp_path):
    ctx = RunContext(user_id="u")
    async with agent(tmp_path, ScriptedLLM(pay_calls(1, 2), say("listo"))) as a:
        paused = await a.run("paga 1 y 2", run_context=ctx)
        first, second = paused.pending_approvals
        first.approve()
        second.reject("demasiado")
        await a.run(run_context=ctx)
    assert PAID == [1]
    denied = ctx.tool_state.records["pay-2"].result
    assert denied is not None and denied.success is False


async def test_a_question_to_the_user_pauses_the_run(tmp_path):
    ask = AssistantMessage(source="llm", content="", tool_calls=[ToolCall(
        id="q1", tool_name="ask_user",
        parameters={"questions": [{"question": "¿Cuánto pago?", "header": "Monto"}]})])
    async with agent(tmp_path, ScriptedLLM(ask)) as a:
        response = await a.run("paga", run_context=RunContext(user_id="u"))
    assert response.finish_reason == "input_needed"


# -------- RESPONSE -----------------------------------------------------------
def test_the_final_message_skips_interim_answers_and_tool_call_shells():
    ctx = RunContext(user_id="u", messages=[
        UserMessage(source="u", content="hola"),
        AssistantMessage(source="a", content="vetada", interim=True),
        AssistantMessage(source="a", content="", tool_calls=[ToolCall(id="c", tool_name="pay", parameters={})]),
        AssistantMessage(source="a", content="la respuesta"),
    ])
    response = AgentResponse(context=ctx, source="a", usage=Usage(), finish_reason="stop")
    assert response.final_text == "la respuesta"
