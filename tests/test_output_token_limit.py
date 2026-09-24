"""Output is capped at 32K and compaction always reserves it in the window."""

from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.compaction.budget import (
    client_max_output_tokens,
    live_message_capacity_tokens,
)
from max_ai.core.model.llm import MAX_OUTPUT_TOKENS, MIN_CONTEXT_WINDOW, ModelConfig


def test_output_is_capped_everywhere(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert OpenAIChatCompletionClient(model="m", max_tokens=100_000).generation_options["max_tokens"] == MAX_OUTPUT_TOKENS
    assert OpenAIChatCompletionClient(model="m", max_tokens=4_000).generation_options["max_tokens"] == 4_000
    assert ModelConfig(max_output_tokens=64_000).max_output_tokens == MAX_OUTPUT_TOKENS

    class CustomClient:  # a client that ignores the cap still gets it reserved capped
        generation_options = {"max_tokens": 500_000}

    assert client_max_output_tokens(CustomClient()) == MAX_OUTPUT_TOKENS


def test_the_smallest_window_still_has_room_for_messages():
    capacity = live_message_capacity_tokens(
        MIN_CONTEXT_WINDOW, max_output_tokens=MAX_OUTPUT_TOKENS, prompt_tokens=10_000,
    )
    assert capacity > MIN_CONTEXT_WINDOW // 2
