"""Agents share a model client safely: the last one to close releases its
connections, and the client stays usable afterwards."""

from __future__ import annotations

from max_ai.agents import Agent
from max_ai.capabilities.clients.ollama.client import OllamaChatCompletionClient
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.workspace.local import LocalWorkspace


def agent(tmp_path, client, name="a"):
    return Agent(name=name, description="d", instructions="i", client=client,
                 workspace=LocalWorkspace(root=tmp_path))


async def test_a_shared_client_closes_with_the_last_agent(tmp_path):
    client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="x")
    pool = client.client
    first, second = agent(tmp_path, client), agent(tmp_path, client, "b")
    await first.close()
    await first.close()  # closing twice gives up its use once
    assert not pool.is_closed()  # the other agent still uses it
    await second.close()
    assert pool.is_closed()
    assert not client.client.is_closed()  # a later agent can use it again


async def test_an_agents_own_client_closes_with_it(tmp_path):
    client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="x")
    pool = client.client
    async with agent(tmp_path, client):
        pass
    assert pool.is_closed()


async def test_ollama_releases_its_pool_and_stays_usable():
    client = OllamaChatCompletionClient(model="qwen3:4b")
    pool = client.client._client
    await client.aclose()
    assert pool.is_closed and not client.client._client.is_closed
