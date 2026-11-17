"""The client contract: messages in, an answer out, no run context needed."""

import pytest

from max_ai.base.clients import CoreChatCompletionClient
from max_ai.core.messages import AssistantMessage, SystemMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError
from max_ai.types.completions import ChatCompletionChunk, ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import CoreToolDefinition


class Echo(CoreChatCompletionClient):
    """A provider that answers with what it was sent."""

    API_KEY_ENV = "ECHO_API_KEY"

    def format_messages(self, ctx, prompts):
        return [SystemMessage(source="system", content=prompts), *ctx.messages]

    def build_api_messages(self, messages):
        return [{"role": m.role, "content": m.content} for m in messages]

    def build_tool_schema(self, tools):
        return [{"name": tool.name} for tool in tools]

    def build_thinking(self, effort):
        return {} if effort is None else {"think": effort}

    def normalize_usage_stats(self, usage):
        return Usage()

    failures: list[Exception] = []  # raised one per call, before answering
    calls = 0

    def map_error(self, error):
        if isinstance(error, ClientError):
            return error
        return ClientError.rate_limit_exceeded(0) if str(error) == "busy" else ClientError.invalid_request(str(error))

    def fail_if_told(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)

    async def complete(self, messages, tools, output_format, effort):
        self.fail_if_told()
        self.sent = {"messages": messages, "tools": tools, **self.build_thinking(effort)}
        said = AssistantMessage(source="echo", content=messages[-1]["content"])
        return ChatCompletionResult(message=said, usage=Usage(), model=self.model, finish_reason="stop")

    async def complete_stream(self, messages, tools, output_format, effort):
        self.fail_if_told()
        for word in messages[-1]["content"].split():
            yield ChatCompletionChunk(content=word, is_complete=False)
            if word == "boom":
                raise RuntimeError("busy")

    def _to_config(self):
        raise NotImplementedError

    @classmethod
    def _from_config(cls, config):
        raise NotImplementedError


WEATHER = CoreToolDefinition(name="weather", description="Forecast for a city")


async def test_it_answers_messages_without_a_run_context():
    client = Echo("echo-1")
    result = await client.run([SystemMessage(source="s", content="Be brief."), UserMessage(source="u", content="hola")])
    assert result.message.content == "hola"
    assert [m["role"] for m in client.sent["messages"]] == ["system", "user"]


async def test_the_loop_lets_the_client_arrange_the_conversation():
    client = Echo("echo-1")
    ctx = RunContext(user_id="u", messages=[UserMessage(source="u", content="hola")])
    result = await client.run(client.format_messages(ctx, "Be brief."), tools=[WEATHER])
    assert result.message.content == "hola" and client.sent["tools"] == [{"name": "weather"}]


async def test_it_streams_the_answer_piece_by_piece():
    chunks = Echo("echo-1").stream([UserMessage(source="u", content="uno dos tres")])
    assert [chunk.content async for chunk in chunks] == ["uno", "dos", "tres"]


async def test_the_effort_comes_from_the_model_config():
    client = Echo("echo-1", config=ModelConfig(supports_thinking=True, effort="high"))
    await client.run([UserMessage(source="u", content="hola")])
    assert client.sent["think"] == "high"
    plain = Echo("echo-1")
    await plain.run([UserMessage(source="u", content="hola")])
    assert "think" not in plain.sent


def test_the_key_is_given_by_hand_or_read_from_the_variable(monkeypatch):
    monkeypatch.setenv("ECHO_API_KEY", "from-env")
    monkeypatch.setenv("OTHER_KEY", "from-other")
    assert Echo("echo-1").api_key == "from-env"
    assert Echo("echo-1", api_key="by-hand").api_key == "by-hand"
    assert Echo("echo-1", api_key_env="OTHER_KEY").api_key == "from-other"


async def test_tools_for_a_model_without_them_fail_before_the_call():
    client = Echo("echo-1", config=ModelConfig(supports_function_calling=False))
    with pytest.raises(ClientError):
        await client.run([UserMessage(source="u", content="hola")], tools=[WEATHER])


HOLA = [UserMessage(source="u", content="hola")]


async def test_a_passing_failure_is_tried_again_three_times_by_default():
    client = Echo("echo-1")
    client.failures = [RuntimeError("busy"), RuntimeError("busy")]
    seen = []
    result = await client.run(HOLA, on_retry=lambda attempt, error, wait: seen.append((attempt, error.kind)))
    assert result.message.content == "hola" and client.calls == 3
    assert seen == [(1, "rate_limit"), (2, "rate_limit")]
    client.failures, client.calls = [RuntimeError("busy")] * 3, 0
    with pytest.raises(ClientError):
        await client.run(HOLA)
    assert client.calls == 3


async def test_the_policy_given_in_the_call_wins_over_the_clients():
    client = Echo("echo-1", retry=RetryPolicy(attempts=500))
    client.failures = [RuntimeError("busy")] * 10
    with pytest.raises(ClientError):
        await client.run(HOLA, retry=RetryPolicy(attempts=5))
    assert client.calls == 5


async def test_a_failure_that_will_repeat_is_not_tried_again():
    client = Echo("echo-1")
    client.failures = [RuntimeError("bad request")]
    with pytest.raises(ClientError):
        await client.run(HOLA)
    assert client.calls == 1


async def test_a_stream_is_tried_again_only_before_anything_came_out():
    client = Echo("echo-1")
    client.failures = [RuntimeError("busy")]
    assert [chunk.content async for chunk in client.stream(HOLA)] == ["hola"]
    assert client.calls == 2
    client.calls, got = 0, []
    with pytest.raises(ClientError):
        async for chunk in client.stream([UserMessage(source="u", content="uno boom dos")]):
            got.append(chunk.content)
    assert got == ["uno", "boom"] and client.calls == 1  # not repeated: text was already out


def test_the_wait_is_read_from_the_standard_headers():
    read = Echo.retry_after
    assert read({"retry-after-ms": "1500"}) == 1.5
    assert read({"retry-after": "7"}) == 7.0
    assert 0 <= read({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) < 1  # a date in the past
    assert read({}) is None and read(None) is None
