"""Guard: serialized components name the env vars that hold secrets, never the
secrets themselves (agent configs are meant to be stored in a database)."""

from __future__ import annotations

import json

import pytest

from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.mcp import (
    HTTPSecret,
    HTTPServerConfig,
    StdioMCPServerConfig,
    StdioSecret,
    serialize_mcp_servers,
)
from max_ai.capabilities.sandbox.local.layout import LocalSandbox
from max_ai.capabilities.session_store import MongoDBSessionStore

SECRETS = {
    "OPENAI_API_KEY": "sk-openai-SECRET",
    "OPENROUTER_API_KEY": "sk-or-SECRET",
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
    lambda: MongoDBSessionStore(),
    lambda: MongoDBKnowledgeRegistry(name="docs", description="d"),
])
def test_components_store_env_var_names(component):
    instance = component()
    payload = instance.serialize().model_dump_json()
    assert_clean(payload)
    restored = type(instance).deserialize(json.loads(payload))  # still usable
    key = getattr(restored, "api_key", None)
    if key is not None:  # read back from the env var: a str or a SecretStr
        assert getattr(key, "get_secret_value", lambda: key)() in SECRETS.values()


def test_mcp_servers_store_env_var_names():
    payload = serialize_mcp_servers([
        HTTPServerConfig(server_id="a", url="https://x/mcp",
                         secret=HTTPSecret(type="bearer", keys={"MCP_TOKEN": SECRETS["MCP_TOKEN"]})),
        StdioMCPServerConfig(server_id="b", command="uvx", args=["srv"],
                             secret=StdioSecret(keys={"MCP_KEY": SECRETS["MCP_KEY"]})),
    ])
    assert_clean(payload)


def test_a_bound_sandbox_never_serializes_its_variables(tmp_path):
    box = LocalSandbox(root=str(tmp_path)).bind("alice", {"GITHUB_TOKEN": "gh-SECRET"})
    assert_clean(box.serialize().model_dump_json())
