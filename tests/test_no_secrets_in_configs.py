"""Guard: serialized components name the env vars that hold secrets, never the
secrets themselves (agent configs are meant to be stored in a database)."""

from __future__ import annotations

import json

import pytest

from max_ai.capabilities.clients import (
    OllamaChatCompletionClient,
    OpenAIChatCompletionClient,
    OpenRouterChatCompletionClient,
)
from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.mcp import (
    HTTPServerConfig,
    StdioMCPServerConfig,
    serialize_mcp_servers,
)
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry

SECRETS = {
    "OPENAI_API_KEY": "sk-openai-SECRET",
    "OPENROUTER_API_KEY": "sk-or-SECRET",
    "OLLAMA_API_KEY": "ollama-SECRET",
    "MONGODB_URI": "mongodb://user:pw-SECRET@host/",
    "MCP_TOKEN": "mcp-token-SECRET",
    "MCP_KEY": "mcp-key-SECRET",
}


@pytest.fixture(autouse=True)
def secrets_in_env(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)


def assert_clean(payload: str) -> None:
    assert "SECRET" not in payload and "*****" not in payload


@pytest.mark.parametrize("component", [
    lambda: OpenAIChatCompletionClient(model="m"),
    lambda: OpenAIChatCompletionClient(model="m", api_key="sk-literal-SECRET"),
    lambda: OpenRouterChatCompletionClient(model="m:free"),
    lambda: OllamaChatCompletionClient(model="m"),
    lambda: MongoDBMemoryRegistry(),
    lambda: MongoDBKnowledgeRegistry(name="docs", description="d"),
])
def test_components_store_env_var_names(component):
    instance = component()
    payload = instance.serialize().model_dump_json()
    assert_clean(payload)
    restored = type(instance).deserialize(json.loads(payload))  # still usable
    key = getattr(restored, "api_key", None)
    if key is not None:
        assert key.get_secret_value() in SECRETS.values()


def test_mcp_servers_store_env_var_names():
    payload = serialize_mcp_servers([
        HTTPServerConfig(server_id="a", url="https://x/mcp", token_env="MCP_TOKEN",
                         headers_env={"X-Key": "MCP_KEY"}),
        StdioMCPServerConfig(server_id="b", command="uvx", args=["srv"],
                             env_from={"API_KEY": "MCP_KEY"}),
    ])
    assert_clean(payload)
