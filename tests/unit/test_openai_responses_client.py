"""The OpenAI Responses client: what it sends, how it reads the answer, and the
reasoning it must give back."""

import pytest
from pydantic import BaseModel

from max_ai.base.component import ComponentBase
from max_ai.capabilities.clients.openai import OpenAIResponsesClient
from max_ai.core.messages import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.errors.client import ClientError
from max_ai.types.tools import CoreToolDefinition

HOLA = [UserMessage(source="u", content="hola")]
WEATHER = CoreToolDefinition(
    name="weather", description="Forecast for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
REASONING = {
    "type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA-secret", "status": "completed",
    "summary": [{"type": "summary_text", "text": "I need the weather."}],
}
CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "weather", "arguments": '{"city": "Lima"}'}


class City(BaseModel):
    name: str


def said(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def answer(*output, status="completed", **more):
    usage = {"input_tokens": 10, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 6}}
    return {"status": status, "output": list(output), "model": "gpt-real", "usage": usage, **more}


def client(*replies, **given) -> OpenAIResponsesClient:
    """A client whose SDK call answers with ``replies`` in order and keeps what it was sent."""
    made = OpenAIResponsesClient("gpt-test", api_key="sk-test", **given)
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

    made.client.responses.create = create
    return made


async def test_the_request_is_in_the_responses_format_and_nothing_is_stored():
    made = client(answer(said("hi")), config=ModelConfig(thinking={"reasoning": {"effort": "low"}}))
    await made.run([SystemMessage(source="s", content="Be brief."), *HOLA], [WEATHER], City)
    sent = made.sent[0]
    assert sent["input"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hola"}]
    assert sent["store"] is False and sent["include"] == ["reasoning.encrypted_content"]
    assert sent["tools"][0] == {
        "type": "function", "name": "weather", "description": "Forecast for a city",
        "parameters": WEATHER.parameters, "strict": False,
    }
    assert sent["text"]["format"]["type"] == "json_schema" and sent["text"]["format"]["name"] == "City"
    assert sent["extra_body"] == {"reasoning": {"effort": "low"}}


async def test_the_answer_is_read_with_its_reasoning_tool_calls_and_usage():
    result = await client(answer(REASONING, CALL)).run(HOLA, [WEATHER])
    message = result.message
    assert message.tool_calls[0].id == "call_1" and message.tool_calls[0].parameters == {"city": "Lima"}
    assert message.thinking == "I need the weather." and result.finish_reason == "tool_calls"
    # What goes back: the id, the summary and the encrypted content, nothing else.
    assert message.reasoning_state == {"responses": [{k: v for k, v in REASONING.items() if k != "status"}]}
    assert (result.usage.tokens_input, result.usage.tokens_output, result.usage.tokens_cached) == (10, 4, 6)


async def test_the_reasoning_is_sent_back_before_what_the_model_said_and_asked_for():
    state = {"responses": [{k: v for k, v in REASONING.items() if k != "status"}]}
    history = [
        *HOLA,
        AssistantMessage(
            source="llm", content="Checking.", reasoning_state=state,
            tool_calls=[ToolCall(id="call_1", tool_name="weather", parameters={"city": "Lima"})],
        ),
        ToolMessage(source="weather", content="sunny", tool_call_id="call_1", tool_name="weather", success=True),
    ]
    made = client(answer(said("sunny in Lima")))
    result = await made.run(history, [WEATHER])
    kinds = [item.get("type") or item["role"] for item in made.sent[0]["input"]]
    assert kinds == ["user", "reasoning", "assistant", "function_call", "function_call_output"]
    assert made.sent[0]["input"][1]["encrypted_content"] == "gAAA-secret"
    assert made.sent[0]["input"][4] == {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}
    assert result.message.text() == "sunny in Lima" and result.finish_reason == "stop"
    # The reasoning of another format is not this client's to send.
    other = AssistantMessage(source="llm", content="x", reasoning_state={"messages": [{"signature": "s"}]})
    assert made.items(other) == [{"role": "assistant", "content": "x"}]


@pytest.mark.parametrize(
    ("details", "finish"),
    [({"reason": "max_output_tokens"}, "length"), ({"reason": "content_filter"}, "content_filter")],
)
async def test_an_answer_that_is_not_whole_says_why_in_our_words(details, finish):
    result = await client(answer(said("cut"), status="incomplete", incomplete_details=details)).run(HOLA)
    assert result.finish_reason == finish


async def test_an_answer_that_failed_is_an_error():
    failed = answer(status="failed", error={"code": "server_error", "message": "boom"})
    with pytest.raises(ClientError, match="server_error: boom"):
        await client(failed, failed, failed).run(HOLA)


async def test_a_stream_gives_thinking_and_text_then_whole_tool_calls_then_the_usage():
    events = [
        {"type": "response.reasoning_summary_text.delta", "delta": "I need "},
        {"type": "response.output_text.delta", "delta": "Chec"},
        {"type": "response.output_text.delta", "delta": "king."},
        {"type": "response.completed", "response": answer(REASONING, said("Checking."), CALL)},
    ]
    chunks = [chunk async for chunk in client(events).stream(HOLA, [WEATHER])]
    assert [c.thinking or c.content for c in chunks[:3]] == ["I need ", "Chec", "king."]
    assert chunks[3].tool_call_chunk["function"] == {"name": "weather", "arguments": '{"city": "Lima"}'}
    last = chunks[-1]
    assert last.is_complete and last.finish_reason == "tool_calls" and last.usage.tokens_input == 10
    assert last.message.text() == "Checking." and last.message.reasoning_state["responses"][0]["id"] == "rs_1"


def test_it_is_stored_without_the_key_and_built_again(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    made = OpenAIResponsesClient("gpt-test", options={"temperature": 0})
    stored = made.serialize().model_dump_json()
    assert "sk-from-env" not in stored
    again = ComponentBase.deserialize(made.serialize().model_dump())
    assert isinstance(again, OpenAIResponsesClient) and again.api_key == "sk-from-env"
    assert again.options == {"temperature": 0}
