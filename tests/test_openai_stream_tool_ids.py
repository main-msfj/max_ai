"""Streamed tool calls get an id even when the provider sends none."""

from __future__ import annotations

import time

from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


def response(tool_call: dict) -> list[dict]:
    """One streamed model response holding a single tool call at index 0."""
    return [{"choices": [{"delta": {"tool_calls": [tool_call]}, "finish_reason": "tool_calls"}]}]


async def tool_call_ids(client: OpenAIChatCompletionClient, chunks: list[dict]) -> list[str]:
    async def stream():
        for chunk in chunks:
            yield chunk

    return [
        chunk.tool_call_chunk["id"]
        async for chunk in client._iter_chunks(stream(), None, time.monotonic())
        if chunk.tool_call_chunk
    ]


async def test_missing_ids_do_not_repeat_across_responses():
    # vLLM and some OpenRouter models send no id; index restarts at 0 on every
    # response, so an id built from it collided in the run's ToolState.
    client = OpenAIChatCompletionClient(model="m", api_key="sk-test")
    call = {"index": 0, "function": {"name": "bash", "arguments": "{}"}}

    first = await tool_call_ids(client, response(call))
    second = await tool_call_ids(client, response(call))

    assert len(first) == len(second) == 1
    assert first[0] and second[0]
    assert first[0] != second[0]


async def test_provider_id_is_kept():
    client = OpenAIChatCompletionClient(model="m", api_key="sk-test")
    call = {"index": 0, "id": "call_abc", "function": {"name": "bash", "arguments": "{}"}}

    assert await tool_call_ids(client, response(call)) == ["call_abc"]
