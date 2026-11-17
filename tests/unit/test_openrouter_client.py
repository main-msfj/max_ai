"""The OpenRouter client: what it adds to the request, the thinking it reads, and errors inside an answer."""

import pytest

from max_ai.base.component import ComponentBase
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.core.messages import UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.retry import RetryPolicy
from max_ai.errors.client import ClientError

HOLA = [UserMessage(source="u", content="hola")]


def answer(content="hi", reasoning=None):
    message = {"content": content, "tool_calls": None, "reasoning": reasoning}
    return {"choices": [{"message": message, "finish_reason": "stop"}], "model": "real", "usage": None}


def client(*replies, **given) -> OpenRouterChatCompletionClient:
    """A client whose SDK call answers with ``replies`` in order and keeps what it was sent."""
    made = OpenRouterChatCompletionClient("openai/gpt-test", api_key="sk-test", **given)
    made.sent = []

    async def create(**request):
        made.sent.append(request)
        reply = replies[len(made.sent) - 1]
        if isinstance(reply, list):
            async def pieces():
                for piece in reply:
                    yield piece
            return pieces()
        return reply

    made.client.chat.completions.create = create
    return made


def test_it_talks_to_openrouter_and_names_the_app():
    made = client(app_name="max", app_url="https://max.example")
    assert str(made.client.base_url) == "https://openrouter.ai/api/v1/"
    headers = made.client.default_headers
    assert headers["X-Title"] == "max" and headers["HTTP-Referer"] == "https://max.example"


async def test_fallbacks_routing_thinking_and_options_go_in_the_body():
    made = client(
        answer(), fallback_models=["anthropic/other"], provider={"order": ["openai"]},
        config=ModelConfig(thinking={"reasoning": {"effort": "low"}}), options={"temperature": 0.2},
    )
    await made.run(HOLA)
    assert made.sent[0]["model"] == "openai/gpt-test"
    assert made.sent[0]["extra_body"] == {
        "models": ["openai/gpt-test", "anthropic/other"], "provider": {"order": ["openai"]},
        "reasoning": {"effort": "low"}, "temperature": 0.2,
    }


async def test_a_plain_client_adds_nothing_to_the_request():
    made = client(answer())
    await made.run(HOLA)
    assert "extra_body" not in made.sent[0]


async def test_the_thinking_text_is_read_from_the_answer_and_from_the_stream():
    result = await client(answer("4", reasoning="2 + 2")).run(HOLA)
    assert (result.message.thinking, result.message.content) == ("2 + 2", "4")
    pieces = [
        {"choices": [{"delta": {"reasoning": "2 + 2"}}]},
        {"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]},
    ]
    chunks = [chunk async for chunk in client(pieces).stream(HOLA)]
    assert (chunks[0].thinking, chunks[1].content) == ("2 + 2", "4")


async def test_an_error_inside_an_answer_is_raised_with_the_providers_words():
    broken = {"error": {"code": 502, "message": "Provider returned error", "metadata": {"raw": "overloaded"}}}
    with pytest.raises(ClientError, match="Provider returned error: overloaded"):
        await client(broken).run(HOLA, retry=RetryPolicy(attempts=1))


async def test_a_rate_limit_inside_an_answer_is_tried_again():
    made = client({"error": {"code": 429, "message": "slow down"}}, answer("ok"))
    result = await made.run(HOLA, retry=RetryPolicy(attempts=2, base_delay=0, max_delay=0))
    assert result.message.content == "ok" and len(made.sent) == 2


async def test_an_error_in_the_middle_of_a_stream_is_raised():
    pieces = [{"choices": [{"delta": {"content": "Let "}}]}, {"error": {"code": 502, "message": "cut"}}]
    with pytest.raises(ClientError, match="cut"):
        [chunk async for chunk in client(pieces).stream(HOLA)]


def test_it_is_stored_without_the_key_and_built_again(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    made = OpenRouterChatCompletionClient(
        "openai/gpt-test", fallback_models=["a/b"], provider={"sort": "price"}, app_name="max",
    )
    stored = made.serialize()
    assert "sk-or" not in stored.model_dump_json() and stored.config["api_key_env"] == "OPENROUTER_API_KEY"
    again = ComponentBase.deserialize(stored.model_dump())
    assert isinstance(again, OpenRouterChatCompletionClient)
    assert (again.api_key, again.fallback_models, again.provider, again.app_name) == (
        "sk-or", ["a/b"], {"sort": "price"}, "max",
    )


def test_without_a_key_it_cannot_be_built(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterChatCompletionClient("openai/gpt-test")
