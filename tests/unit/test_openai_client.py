"""The OpenAI client: what it sends, how it reads the answer, and its errors."""

import httpx
import pytest
from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel

from max_ai.base.component import ComponentBase
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.messages import (
    AssistantMessage,
    ImagePart,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError
from max_ai.types.run_context import RunContext
from max_ai.types.tools import CoreToolDefinition

HOLA = [UserMessage(source="u", content="hola")]
WEATHER = CoreToolDefinition(
    name="weather", description="Forecast for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


class City(BaseModel):
    name: str
    country: str = "PE"


def answer(content="hi", tool_calls=None, usage=None):
    message = {"content": content, "tool_calls": tool_calls}
    return {"choices": [{"message": message, "finish_reason": "stop"}], "model": "gpt-real", "usage": usage}


def failure(kind, status, headers=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return kind("failed", response=httpx.Response(status, headers=headers, request=request), body=None)


def client(*replies, **given) -> OpenAIChatCompletionClient:
    """A client whose SDK call answers with ``replies`` in order and keeps what it was sent."""
    made = OpenAIChatCompletionClient("gpt-test", api_key="sk-test", **given)
    made.sent = []

    async def create(**request):
        made.sent.append(request)
        reply = replies[len(made.sent) - 1]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            async def pieces():
                for piece in reply:
                    yield piece
            return pieces()
        return reply

    made.client.chat.completions.create = create
    return made


async def test_the_request_carries_messages_tools_and_the_models_limits():
    made = client(answer(), config=ModelConfig(max_output_tokens=500))
    conversation = [
        SystemMessage(source="s", content="Be brief."),
        UserMessage(source="u", content="weather in Lima?"),
        AssistantMessage(source="llm", content="", tool_calls=[ToolCall(id="c1", tool_name="weather", parameters={"city": "Lima"})]),
        ToolMessage(source="tool", tool_call_id="c1", tool_name="weather", content="sunny", success=True),
    ]
    await made.run(conversation, tools=[WEATHER])
    sent = made.sent[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["messages"][2]["tool_calls"][0]["function"] == {"name": "weather", "arguments": '{"city": "Lima"}'}
    assert sent["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "sunny"}
    assert sent["tools"][0]["function"]["name"] == "weather" and sent["tool_choice"] == "auto"
    assert sent["max_completion_tokens"] == 500 and sent["stream"] is False
    assert "extra_body" not in sent  # no thinking given: the provider's default


async def test_thinking_and_options_reach_the_request_as_they_are():
    config = ModelConfig(thinking={"reasoning_effort": "low"})
    made = client(answer(), config=config, options={"temperature": 0.2})
    await made.run(HOLA)
    assert made.sent[0]["extra_body"] == {"reasoning_effort": "low", "temperature": 0.2}


async def test_the_answer_is_read_with_its_tool_calls_and_usage():
    calls = [
        {"id": "c1", "function": {"name": "weather", "arguments": '{"city": "Lima"}'}},
        {"id": "c2", "function": {"name": "weather", "arguments": "{not json"}},
    ]
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80}}
    result = await client(answer("", calls, usage)).run(HOLA, tools=[WEATHER])
    first, second = result.message.tool_calls
    assert (first.id, first.tool_name, first.parameters) == ("c1", "weather", {"city": "Lima"})
    assert second.parameters == {"raw_error_content": "{not json", "parsing_error": True}
    assert (result.usage.tokens_input, result.usage.tokens_output, result.usage.tokens_cached) == (100, 20, 80)
    assert result.usage.llm_calls == 1 and result.usage.tool_calls == 2 and result.model == "gpt-real"


async def test_a_structured_answer_comes_back_as_the_model():
    made = client(answer('{"name": "Lima", "country": "PE"}'))
    result = await made.run(HOLA, output_format=City)
    assert result.message.structured_output == City(name="Lima")
    schema = made.sent[0]["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["required"] == ["name", "country"]
    assert schema["schema"]["additionalProperties"] is False
    assert "default" not in schema["schema"]["properties"]["country"]


async def test_an_image_needs_a_model_that_sees():
    picture = [UserMessage(source="u", content=[TextPart(text="what is it?"), ImagePart(data=b"png", mime_type="image/png")])]
    seeing = client(answer(), config=ModelConfig(supports_vision=True))
    await seeing.run(picture)
    parts = seeing.sent[0]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is it?"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,cG5n"
    with pytest.raises(ClientError):
        await client(answer()).run(picture)


async def test_a_stream_gives_the_text_then_whole_tool_calls_then_the_usage():
    pieces = [
        {"choices": [{"delta": {"content": "Let "}}]},
        {"choices": [{"delta": {"content": "me check."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "weather", "arguments": '{"ci'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Lima"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
    ]
    made = client(pieces)
    chunks = [chunk async for chunk in made.stream(HOLA, tools=[WEATHER])]
    assert [c.content for c in chunks[:2]] == ["Let ", "me check."]
    assert chunks[2].tool_call_chunk == {"id": "c1", "function": {"name": "weather", "arguments": '{"city": "Lima"}'}}
    last = chunks[-1]
    assert last.is_complete and last.finish_reason == "tool_calls"
    assert (last.usage.tokens_input, last.usage.tokens_output, last.usage.tool_calls) == (9, 4, 1)
    assert made.sent[0]["stream"] is True and made.sent[0]["stream_options"] == {"include_usage": True}


async def test_a_rate_limit_waits_what_the_provider_asked_and_tries_again():
    made = client(failure(RateLimitError, 429, {"retry-after-ms": "10"}), answer("ok"))
    waits = []
    result = await made.run(HOLA, on_retry=lambda attempt, error, wait: waits.append((error.kind, wait)))
    assert result.message.content == "ok" and waits == [("rate_limit", 0.01)]


async def test_a_wrong_key_fails_at_once_as_a_client_error():
    made = client(failure(AuthenticationError, 401), answer("never"))
    with pytest.raises(ClientError) as caught:
        await made.run(HOLA, retry=RetryPolicy(attempts=5))
    assert not caught.value.transient
    assert len(made.sent) == 1


def test_the_loop_gets_the_system_prompt_first_then_the_conversation():
    made = client()
    ctx = RunContext(user_id="u", messages=HOLA)
    roles = [m.role for m in made.format_messages(ctx, "You are brief.")]
    assert roles == ["system", "user"]
    assert [m.role for m in made.format_messages(ctx, "  ")] == ["user"]


def test_it_is_stored_without_the_key_and_built_again(monkeypatch):
    monkeypatch.setenv("TEAM_KEY", "sk-team")
    made = OpenAIChatCompletionClient(
        "gpt-test", api_key_env="TEAM_KEY", options={"temperature": 0.1}, retry=RetryPolicy(attempts=5),
    )
    stored = made.serialize()
    assert "sk-team" not in stored.model_dump_json() and stored.config["api_key_env"] == "TEAM_KEY"
    again = ComponentBase.deserialize(stored.model_dump())
    assert isinstance(again, OpenAIChatCompletionClient)
    assert (again.api_key, again.options, again.retry.attempts) == ("sk-team", {"temperature": 0.1}, 5)


def test_without_a_key_it_cannot_be_built(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatCompletionClient("gpt-test")
