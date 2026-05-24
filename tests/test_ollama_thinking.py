"""Tests for Ollama thinking request configuration."""

from __future__ import annotations

import pytest

from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.models import ModelConfig, OllamaChatCompletionClientConfig
from max_ai.errors.client import ClientError


def _client(think=None) -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model="qwen3",
        host="http://localhost:11434",
        config=ModelConfig(supports_thinking=True),
        think=think,
    )


@pytest.mark.parametrize("think", [True, False, "low", "medium", "high"])
def test_build_request_accepts_ollama_thinking_modes(think):
    client = _client(think=think)

    request = client._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        output_format=None,
        stream=False,
    )

    assert request["think"] == think


def test_per_call_think_overrides_client_default():
    client = _client(think=True)

    request = client._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        output_format=None,
        stream=False,
        think="high",
    )

    assert request["think"] == "high"


def test_build_request_omits_think_when_unset():
    client = _client()

    request = client._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        output_format=None,
        stream=False,
    )

    assert "think" not in request


@pytest.mark.parametrize("think", ["off", "minimal", 1])
def test_invalid_ollama_think_value_raises(think):
    with pytest.raises(ClientError, match="Ollama think must"):
        _client(think=think)


def test_config_schema_accepts_gpt_oss_effort():
    config = OllamaChatCompletionClientConfig(
        model="gpt-oss",
        host="http://localhost:11434",
        think="medium",
    )

    assert config.think == "medium"


def test_max_tokens_is_translated_to_ollama_num_predict():
    client = OllamaChatCompletionClient(
        model="qwen3",
        host="http://localhost:11434",
        max_tokens=123,
    )

    assert client.generation_options == {"max_tokens": 123}

    request = client._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        output_format=None,
        stream=False,
    )

    assert request["options"] == {"num_predict": 123}


def test_per_call_max_tokens_overrides_client_default():
    client = OllamaChatCompletionClient(
        model="qwen3",
        host="http://localhost:11434",
        max_tokens=456,
    )

    request = client._build_request(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        output_format=None,
        stream=False,
        max_tokens=789,
    )

    assert request["options"] == {"num_predict": 789}
