"""The client contract: messages in, an answer out, no run context needed."""

import pytest

from max_ai.base.clients import CoreLLMClient
from max_ai.core.messages import AssistantMessage, SystemMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError
from max_ai.types.completions import LLMChunk, LLMResponse, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import CoreToolDefinition


class Echo(CoreLLMClient):
    """A provider that answers with what it was sent."""

    API_KEY_ENV = "ECHO_API_KEY"

    def format_messages(self, ctx, prompts):
        return [SystemMessage(source="system", content=prompts), *ctx.messages]

    def build_api_messages(self, messages):
        return [{"role": m.role, "content": m.content} for m in messages]

    def build_tool_schema(self, tools):
        return [{"name": tool.name} for tool in tools]

    def normalize_usage_stats(self, usage):
        return Usage()

    failures: list[Exception] = []  # raised one per call, before answering
    calls = 0

    def blocked(self, error):
        return ["bad words"] if str(error) == "blocked" else None

    def window_full(self, error):
        return str(error) == "too long"

    def rate_limited(self, error):
        return str(error) == "busy"

    def wait_asked(self, error):
        return 0

    def other_error(self, error):
        return ClientError.invalid_request(str(error))

    def fail_if_told(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)

    async def complete(self, messages, tools, output_format):
        self.fail_if_told()
        self.sent = {"messages": messages, "tools": tools, **(self.config.thinking or {})}
        said = AssistantMessage(source="echo", content=messages[-1]["content"])
        return LLMResponse(message=said, usage=Usage(), model=self.model, finish_reason="stop")

    async def complete_stream(self, messages, tools, output_format):
        self.fail_if_told()
        for word in messages[-1]["content"].split():
            yield LLMChunk(content=word, is_complete=False)
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


async def test_thinking_comes_from_the_model_config_as_it_is():
    client = Echo("echo-1", config=ModelConfig(thinking={"think": "high"}))
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


class Refused(Exception):
    """A provider's error as its SDK raises it: a status, a code and the body."""

    def __init__(self, text: str, status_code: int, code: str | None = None, body=None) -> None:
        super().__init__(text)
        self.status_code, self.code, self.body = status_code, code, body


def test_the_parent_maps_and_asks_each_provider_for_the_cases_it_reacts_to():
    client = Echo("echo-1")
    kinds = {text: client.map_error(RuntimeError(text)).kind for text in ("blocked", "too long", "busy", "x")}
    assert kinds == {
        "blocked": "content_filter", "too long": "token_limit", "busy": "rate_limit",
        "x": "invalid_request",
    }
    blocked = client.map_error(RuntimeError("blocked"))
    assert blocked.found == ["bad words"] and not blocked.transient


def test_a_client_that_does_not_say_how_it_is_blocked_cannot_be_created():
    class Lazy(Echo):
        blocked = CoreLLMClient.blocked  # left as the parent's: abstract

    with pytest.raises(TypeError, match="blocked"):
        Lazy("lazy-1")
    # The class of the OpenAI format leaves it to each service on purpose.
    from max_ai.capabilities.clients._chat_completions import ChatCompletionsClient

    assert "blocked" in ChatCompletionsClient.__abstractmethods__


# One refused request per client of the package, as its provider sends it.
REFUSED = {
    "max_ai.capabilities.clients.openrouter.OpenRouterChatCompletionClient": Refused(
        "Error code: 403 - Request blocked: prompt injection patterns detected", 403,
        body={"message": "Request blocked", "code": 403, "metadata": {"patterns": ["reveal_prompt"]}},
    ),
    "max_ai.capabilities.clients.openai.OpenAIChatCompletionClient": Refused(
        "Error code: 400 - The response was filtered", 400, code="content_filter",
    ),
    "max_ai.capabilities.clients.openai.OpenAIResponsesClient": Refused(
        "Error code: 400 - The response was filtered", 400, code="content_filter",
    ),
    "max_ai.capabilities.clients.anthropic.AnthropicClient": Refused(
        "Error code: 400 - Output blocked by content filtering policy", 400,
    ),
}


def test_every_client_of_the_package_tells_a_refused_request_apart():
    import importlib
    import pkgutil

    from max_ai.capabilities import clients
    from max_ai.capabilities.clients._chat_completions import ChatCompletionsClient

    # Every client the package has, found by itself: a new one must add its example.
    for found in pkgutil.iter_modules(clients.__path__):
        importlib.import_module(f"{clients.__name__}.{found.name}")
    from max_ai.capabilities.clients._messages import MessagesClient
    from max_ai.capabilities.clients._responses import ResponsesClient

    services = [
        *ChatCompletionsClient.__subclasses__(), *ResponsesClient.__subclasses__(),
        *MessagesClient.__subclasses__(),
    ]
    assert {f"{cls.__module__.rsplit('.', 1)[0]}.{cls.__name__}" for cls in services} == set(REFUSED)
    for cls in services:
        client = cls("some/model", api_key="sk-test")
        error = client.map_error(REFUSED[f"{cls.__module__.rsplit('.', 1)[0]}.{cls.__name__}"])
        assert error.kind == "content_filter", cls.__name__
        # Another 403 or 400 is not taken for one.
        assert client.map_error(Refused("Error code: 403 - no credits", 403)).kind != "content_filter"
