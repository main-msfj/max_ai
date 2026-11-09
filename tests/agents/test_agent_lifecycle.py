"""The Agent's run lifecycle: answer, stream, fail, cancel, pause for
approvals and questions, and resume with the user's decisions."""

from __future__ import annotations

import asyncio

import pytest

from max_ai.agents import Agent, Policy
from max_ai.base.middleware import CoreMiddleware
from max_ai.capabilities.tools.ask_user import AskUserTool
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.event_type import ErrorEvent, ModelCallEvent, ModelResponseEvent
from max_ai.core.harness import messages as harness
from max_ai.core.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.termination import CancellationToken
from max_ai.types.agent_response import AgentResponse
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolResult
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


def agent(tmp_path, llm, **options) -> Agent:
    tool = FunctionAsTool(pay, approval_mode=ToolApprovalMode.ASK_APPROVED)
    a = Agent(name="a", description="d", instructions="i", client=llm, toolset=[tool],
              workspace=LocalWorkspace(root=tmp_path), **options)
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


def ask_call() -> AssistantMessage:
    return AssistantMessage(source="llm", content="", tool_calls=[ToolCall(
        id="q1", tool_name=AskUserTool.TOOL_NAME,
        parameters={"questions": [{"question": "¿Cuánto pago?", "header": "Monto"}]})])


class SpyMiddleware(CoreMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []
        self.responses: list[ToolResult] = []

    async def on_tool_request(self, mw, request):
        self.requests.append(request.record.id)

    async def on_tool_response(self, mw, request, result):
        self.responses.append(result)
        return result


async def test_a_question_to_the_user_pauses_the_run(tmp_path):
    async with agent(tmp_path, ScriptedLLM(ask_call())) as a:
        response = await a.run("paga", run_context=RunContext(user_id="u"))
    assert response.finish_reason == "input_needed"


async def test_an_answer_survives_a_restart_and_reaches_the_model_once(tmp_path):
    spy = SpyMiddleware()
    async with agent(tmp_path, ScriptedLLM(ask_call(), say("listo")), middlewares=[spy]) as a:
        paused = await a.run("paga", run_context=RunContext(user_id="u"))
        assert spy.requests == []  # the wait happens before the middleware

        ctx = RunContext.model_validate(paused.context.model_dump(mode="json"))
        record = ctx.tool_state.records["q1"]
        ctx.tool_state.provide_result("q1", AskUserTool.answer(record, {"¿Cuánto pago?": "10"}))
        ctx = RunContext.model_validate(ctx.model_dump(mode="json"))
        done = await a.resume(ctx)

    assert done.finish_reason == "stop"
    assert spy.requests == ["q1"]
    assert spy.responses[0].result == {"answers": {"¿Cuánto pago?": "10"}}
    messages = [m for m in done.context.messages if isinstance(m, ToolMessage)]
    assert len(messages) == 1 and "10" in str(messages[0].content)


async def test_resuming_without_an_answer_stays_paused(tmp_path):
    spy = SpyMiddleware()
    async with agent(tmp_path, ScriptedLLM(ask_call()), middlewares=[spy]) as a:
        paused = await a.run("paga", run_context=RunContext(user_id="u"))
        again = await a.resume(paused.context)
    assert again.finish_reason == "input_needed" and spy.requests == []
    assert again.context.tool_state.records["q1"].is_awaiting_input


async def test_a_new_message_closes_an_open_question_first(tmp_path):
    ctx = RunContext(user_id="u")
    async with agent(tmp_path, ScriptedLLM(ask_call(), say("ok"))) as a:
        await a.run("paga", run_context=ctx)
        done = await a.run("mejor otra cosa", run_context=ctx)
    assert done.finish_reason == "stop"
    kinds = [type(m) for m in done.context.messages]
    assert kinds.index(ToolMessage) < len(kinds) - 1 - kinds[::-1].index(UserMessage)
    closed = next(m for m in done.context.messages if isinstance(m, ToolMessage))
    assert closed.success is False and closed.content == harness.QUESTION_NOT_ANSWERED


async def test_a_message_in_the_inbox_is_read_instead_of_pausing(tmp_path):
    ctx = RunContext(user_id="u")

    class WritesMeanwhile(ScriptedLLM):
        async def run(self, **kwargs):
            result = await super().run(**kwargs)
            if len(self.replies) == 1:
                ctx.inbox.append("cambio de planes")
            return result

    async with agent(tmp_path, WritesMeanwhile(ask_call(), say("entendido"))) as a:
        done = await a.run("paga", run_context=ctx)
    assert done.finish_reason == "stop" and not done.needs_input
    texts = [m for m in done.context.messages if isinstance(m, (ToolMessage, UserMessage))]
    assert isinstance(texts[1], ToolMessage) and texts[2].content == "cambio de planes"


async def test_an_open_approval_still_blocks_a_new_message(tmp_path):
    ctx = RunContext(user_id="u")
    async with agent(tmp_path, ScriptedLLM(pay_calls(10))) as a:
        await a.run("paga", run_context=ctx)
        with pytest.raises(ValueError, match="Resolve pending tool calls"):
            await a.run("otra cosa", run_context=ctx)


async def test_a_policy_can_deny_asking_without_looping(tmp_path):
    policy = Policy(deny=[AskUserTool.TOOL_NAME])
    llm = ScriptedLLM(ask_call())
    async with agent(tmp_path, llm, policy=policy) as a:
        response = await a.run("paga", run_context=RunContext(user_id="u"))
    assert response.finish_reason == "tool_denied"
    assert response.context.tool_state.records["q1"].result.success is False


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
