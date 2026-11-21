"""An Agent is storable data: serialize once, deserialize per request."""

from __future__ import annotations

import json
import typing as t

import pytest
from pydantic import BaseModel, Field

from max_ai.agents import Agent
from max_ai.base import component
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.loop import ReactLoop
from max_ai.capabilities.mcp import HTTPSecret, HTTPServerConfig
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.middleware import BudgetMiddleware
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.decorator import tool
from max_ai.core.model.json_schema import model_from_schema


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-stored")
    monkeypatch.setattr(component, "_ALLOWED_PREFIXES", {"max_ai."})


class Item(BaseModel):
    name: str
    qty: int = 1


class Invoice(BaseModel):
    """An invoice."""

    customer: str = Field(description="Who pays.")
    status: t.Literal["paid", "due"]
    items: list[Item]
    note: str | None = None


def full_agent(tmp_path, **extra) -> Agent:
    return Agent(
        "billing", "d", "i", OpenAIChatCompletionClient(model="gpt-4o-mini"),
        sandbox=LocalSandbox(str(tmp_path / "work")),
        memory=Memory(read_only=True),
        reasoning=ReactLoop(max_iterations=12),
        middlewares=[BudgetMiddleware(max_tokens=5000)],
        output_format=Invoice,
        mcp=[HTTPServerConfig(server_id="docs", url="http://localhost:9000/mcp",
                              secret=HTTPSecret(type="bearer", keys={"DOCS_TOKEN": "tok-never-stored"}))],
        **extra,
    )


def test_an_agent_round_trips_through_json(tmp_path):
    row = full_agent(tmp_path).serialize().model_dump_json()
    agent = Agent.deserialize(row)

    assert (agent.name, agent.client.model) == ("billing", "gpt-4o-mini")
    assert isinstance(agent.sandbox, LocalSandbox) and agent.memory is not None and agent.memory.read_only
    assert isinstance(agent.reasoning, ReactLoop) and agent.reasoning.max_iterations == 12
    assert isinstance(agent.middlewares[1], BudgetMiddleware) and agent.middlewares[1].max_tokens == 5000
    assert list(agent.mcp[0].secret.keys) == ["DOCS_TOKEN"] and "tok-never-stored" not in row
    assert agent.serialize().model_dump_json() == row  # stable


def test_the_stored_agent_holds_no_secrets(tmp_path):
    row = full_agent(tmp_path).serialize().model_dump_json()
    assert "sk-never-stored" not in row
    assert '"api_key_env":"OPENAI_API_KEY"' in row


def test_output_format_is_stored_as_json_schema(tmp_path):
    spec = full_agent(tmp_path).serialize().config
    assert spec["output_format"] == Invoice.model_json_schema()

    rebuilt = Agent.deserialize(json.dumps({"provider": "max_ai.agents.Agent", "config": spec}))
    shape = rebuilt.output_format
    assert shape is not None and shape.model_json_schema() == Invoice.model_json_schema()
    ok = t.cast(t.Any, shape.model_validate({"customer": "Ana", "status": "paid", "items": [{"name": "x"}]}))
    assert ok.items[0].qty == 1 and ok.note is None
    with pytest.raises(ValueError):
        shape.model_validate({"customer": "Ana", "status": "lost", "items": []})


def test_schema_rebuild_handles_nesting_enums_and_optionals():
    class Node(BaseModel):
        tags: dict[str, int]
        kind: t.Literal["a", "b"] | None = None
        children: list[Item] = []

    rebuilt = model_from_schema(Node.model_json_schema())
    assert rebuilt.model_json_schema() == Node.model_json_schema()


def test_python_function_tools_cannot_be_stored(tmp_path):
    def lookup(order_id: str) -> str:
        """Find an order by its id."""
        return order_id

    agent = full_agent(tmp_path, toolset=[tool(lookup)])
    with pytest.raises(TypeError, match="MCP server"):
        agent.serialize()
