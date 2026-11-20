"""The Anthropic client: what it sends, how it reads the answer, and the
thinking blocks it must give back."""

import pytest
from anthropic import BadRequestError, OverloadedError
from httpx2 import Request, Response
from pydantic import BaseModel

from max_ai.base.component import ComponentBase
from max_ai.capabilities.clients.anthropic import AnthropicClient
from max_ai.core.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.types.tools import CoreToolDefinition

HOLA = [UserMessage(source="u", content="hola")]
WEATHER = CoreToolDefinition(
    name="weather", description="Forecast for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
THOUGHT = {"type": "thinking", "thinking": "I need the weather.", "signature": "sig-1"}
CALL = {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Lima"}}


class City(BaseModel):
    name: str


def answer(*blocks, stop="end_turn"):
    usage = {"input_tokens": 4, "output_tokens": 5, "cache_read_input_tokens": 6, "cache_creation_input_tokens": 2}
    return {"content": list(blocks), "stop_reason": stop, "model": "claude-real", "usage": usage}


class Stream:
    """The SDK's stream: its events, then the whole message."""

    def __init__(self, events, final):
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *error):
        return False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def get_final_message(self):
        return self.final


def client(*replies, events=(), **given) -> AnthropicClient:
    """A client whose SDK call answers with ``replies`` in order and keeps what it was sent."""
    made = AnthropicClient("claude-test", api_key="sk-test", **given)
    made.sent = []

    def call(request):
        made.sent.append(request)
        reply = replies[len(made.sent) - 1]
        if isinstance(reply, Exception):
            raise reply
        return Stream(events, reply)

    made.call = call
    return made


def failure(cls, status: int, text: str):
    response = Response(status, request=Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls(text, response=response, body=None)


async def test_the_system_prompt_goes_apart_and_the_tools_in_its_format():
    made = client(answer({"type": "text", "text": "hola"}))
    await made.run([SystemMessage(source="s", content="Be brief."), *HOLA], tools=[WEATHER])
    sent = made.sent[0]
    assert sent["system"] == "Be brief."
    assert sent["messages"] == [{"role": "user", "content": "hola"}]
    assert sent["tools"] == [{"name": "weather", "description": "Forecast for a city", "input_schema": WEATHER.parameters}]
    assert sent["max_tokens"] == made.config.max_output_tokens
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert "extra_body" not in sent


async def test_the_results_of_one_round_go_together_in_one_user_message():
    calls = [ToolCall(id="a", tool_name="weather", parameters={"city": "Lima"}),
             ToolCall(id="b", tool_name="weather", parameters={"city": "Cusco"})]
    history = [
        *HOLA,
        AssistantMessage(source="llm", content="", tool_calls=calls),
        ToolMessage(source="t", content="sun", tool_call_id="a", tool_name="weather", success=True),
        ToolMessage(source="t", content="no such city", tool_call_id="b", tool_name="weather", success=False),
        UserMessage(source="u", content="gracias"),
    ]
    made = client(answer({"type": "text", "text": "ok"}))
    await made.run(history)
    turns = made.sent[0]["messages"]
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user", "user"]
    assert turns[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "weather", "input": {"city": "Lima"}},
        {"type": "tool_use", "id": "b", "name": "weather", "input": {"city": "Cusco"}},
    ]
    assert turns[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "sun"},
        {"type": "tool_result", "tool_use_id": "b", "content": "no such city", "is_error": True},
    ]}


async def test_the_answer_is_read_with_its_thinking_tool_calls_and_usage():
    made = client(answer(THOUGHT, {"type": "text", "text": "Let me see."}, CALL, stop="tool_use"))
    result = await made.run(HOLA, tools=[WEATHER])
    message = result.message
    assert (message.content, message.thinking) == ("Let me see.", "I need the weather.")
    assert [(call.id, call.tool_name, call.parameters) for call in message.tool_calls] == [
        ("toolu_1", "weather", {"city": "Lima"})
    ]
    assert result.finish_reason == "tool_calls" and result.model == "claude-real"
    usage = result.usage
    assert (usage.tokens_input, usage.tokens_output, usage.tokens_cached) == (12, 5, 6)


async def test_an_answer_with_thinking_goes_back_block_by_block_as_it_came():
    blocks = [THOUGHT, {"type": "text", "text": "Let me see."}, CALL]
    made = client(answer(*blocks, stop="tool_use"), answer({"type": "text", "text": "sun"}))
    first = await made.run(HOLA, tools=[WEATHER])
    assert first.message.reasoning_state == {"messages": blocks}

    result = ToolMessage(source="t", content="sun", tool_call_id="toolu_1", tool_name="weather", success=True)
    await made.run([*HOLA, first.message, result], tools=[WEATHER])
    assert made.sent[1]["messages"][1] == {"role": "assistant", "content": blocks}
    # An answer without thinking keeps nothing, and one that said nothing is not sent.
    plain = await client(answer({"type": "text", "text": "hola"})).run(HOLA)
    assert plain.message.reasoning_state == {}
    assert made.build_api_messages([AssistantMessage(source="llm", content="")]) == []


@pytest.mark.parametrize(
    ("stop", "finish"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "content_filter"), ("pause_turn", "stop")],
)
async def test_why_the_answer_ended_is_said_in_our_words(stop, finish):
    result = await client(answer({"type": "text", "text": "x"}, stop=stop)).run(HOLA)
    assert result.finish_reason == finish


async def test_a_model_that_thinks_is_asked_for_a_summary_unless_told_how():
    made = client(answer({"type": "text", "text": "x"}), config=ModelConfig(supports_thinking=True))
    await made.run(HOLA)
    assert made.sent[0]["extra_body"] == {"thinking": {"type": "adaptive", "display": "summarized"}}

    told = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    made = client(answer({"type": "text", "text": '{"name": "Lima"}'}), config=ModelConfig(thinking=told))
    result = await made.run(HOLA, output_format=City)
    config = made.sent[0]["extra_body"]["output_config"]
    assert config["effort"] == "low" and config["format"]["type"] == "json_schema"
    assert config["format"]["schema"]["additionalProperties"] is False
    assert result.message.structured_output == City(name="Lima")


async def test_a_stream_gives_thinking_and_text_then_whole_tool_calls_then_the_usage():
    events = [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "I need"}},
        {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "sig-1"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Let me see."}},
        {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
    ]
    made = client(answer(THOUGHT, {"type": "text", "text": "Let me see."}, CALL, stop="tool_use"), events=events)
    chunks = [chunk async for chunk in made.stream(HOLA, tools=[WEATHER])]
    assert [(c.thinking, c.content) for c in chunks[:2]] == [("I need", ""), (None, "Let me see.")]
    assert chunks[2].tool_call_chunk == {"id": "toolu_1", "function": {"name": "weather", "arguments": '{"city": "Lima"}'}}
    last = chunks[-1]
    assert last.is_complete and last.finish_reason == "tool_calls" and len(chunks) == 4
    assert last.message.reasoning_state["messages"][0] == THOUGHT and last.usage.tokens_input == 12


def test_its_errors_are_told_apart():
    made = client()
    full = failure(BadRequestError, 400, "prompt is too long: 250000 tokens > 200000 maximum")
    assert made.map_error(full).kind == "token_limit"
    assert made.map_error(failure(OverloadedError, 529, "Overloaded")).kind == "rate_limit"
    refused = failure(BadRequestError, 400, "Output blocked by content filtering policy")
    assert made.map_error(refused).kind == "content_filter"
    assert made.map_error(failure(BadRequestError, 400, "max_tokens: too large")).kind not in (
        "content_filter", "token_limit", "rate_limit",
    )


def test_it_is_stored_without_the_key_and_built_again(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    made = AnthropicClient("claude-test", options={"speed": "fast"})
    stored = made.serialize().model_dump_json()
    assert "sk-secret" not in stored and "ANTHROPIC_API_KEY" in stored
    again = ComponentBase.deserialize(made.serialize().model_dump())
    assert isinstance(again, AnthropicClient)
    assert (again.model, again.options, again.api_key) == ("claude-test", {"speed": "fast"}, "sk-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient("claude-test")
