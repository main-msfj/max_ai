"""An Agent is storable data: serialize once, deserialize per request."""

from __future__ import annotations

import json
import typing as t

import pytest
from pydantic import BaseModel, Field

from max_ai.agents import Agent
from max_ai.base import component
from max_ai.base.completion_gate import (
    CompletionBase,
    CompletionConfig,
    CompletionDecision,
)
from max_ai.capabilities.clients.openai.client import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.completion_gate import RuntimeGateConfig
from max_ai.capabilities.mcp import HTTPServerConfig
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.reasoning.guards import RepetitionGuard
from max_ai.capabilities.reasoning.react import ReactLoop
from max_ai.capabilities.workspace.local import LocalWorkspace
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
        name="billing", description="d", instructions="i",
        client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        workspace=LocalWorkspace(root=tmp_path / "work"),
        memory=LocalMemoryRegistry(base_path=tmp_path / "data"),
        compaction=SummaryCompaction(threshold=0.8, keep_ratio=0.4),
        reasoning=ReactLoop(max_loop_iterations=12, guards=[RepetitionGuard(max_repeats=4)]),
        completion=RuntimeGateConfig(plan_must_close=False),
        output_format=Invoice,
        mcp=[HTTPServerConfig(server_id="docs", url="http://localhost:9000/mcp", token_env="DOCS_TOKEN")],
        **extra,
    )


def test_an_agent_round_trips_through_json(tmp_path):
    row = full_agent(tmp_path).serialize().model_dump_json()
    agent = Agent.deserialize(row)

    assert (agent.name, agent.client.model) == ("billing", "gpt-4o-mini")
    assert isinstance(agent.memory, LocalMemoryRegistry)
    assert isinstance(agent.compaction, SummaryCompaction)
    assert agent.reasoning.max_loop_iterations == 12
    assert agent.reasoning.guards[0].max_repeats == 4
    assert agent.completion.plan_must_close is False
    assert agent.mcp_servers[0].token_env == "DOCS_TOKEN"
    assert agent.serialize().model_dump_json() == row  # stable


def test_the_stored_agent_holds_no_secrets(tmp_path):
    row = full_agent(tmp_path).serialize().model_dump_json()
    assert "sk-never-stored" not in row
    assert '"api_key_env":"OPENAI_API_KEY"' in row


def test_output_format_is_stored_as_json_schema(tmp_path):
    spec = full_agent(tmp_path).serialize().config
    assert spec["output_format"] == Invoice.model_json_schema()

    rebuilt = Agent.deserialize(json.dumps({"provider": "maxai.agents.Agent", "config": spec}))
    shape = rebuilt.output_format
    assert shape.model_json_schema() == Invoice.model_json_schema()
    ok = shape.model_validate({"customer": "Ana", "status": "paid", "items": [{"name": "x"}]})
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
        """Find an order."""
        return order_id

    agent = full_agent(tmp_path, toolset=[lookup])
    with pytest.raises(TypeError, match="MCP server"):
        agent.serialize()


class ApprovedByFinanceConfig(CompletionConfig):
    min_total: float = 0


class ApprovedByFinance(CompletionBase):
    """A developer's own gate."""

    component_schema = ApprovedByFinanceConfig

    def __init__(self, min_total: float = 0) -> None:
        super().__init__()
        self.min_total = min_total

    def _to_config(self) -> ApprovedByFinanceConfig:
        return ApprovedByFinanceConfig(min_total=self.min_total)

    @classmethod
    def _from_config(cls, config: ApprovedByFinanceConfig) -> "ApprovedByFinance":
        return cls(min_total=config.min_total)

    def on_final_response(self, ctx) -> CompletionDecision:
        return CompletionDecision(status="completed")


class NotStorableGate(CompletionBase):
    def on_final_response(self, ctx) -> CompletionDecision:
        return CompletionDecision(status="completed")


def test_developer_gates_are_stored_by_provider(tmp_path):
    row = full_agent(tmp_path, completion_handlers=[ApprovedByFinance(500)]).serialize().model_dump_json()
    with pytest.raises(PermissionError, match="allow_providers"):
        Agent.deserialize(row)

    component.allow_providers(ApprovedByFinance.__module__ + ".")
    gate = Agent.deserialize(row).completion_handlers[0]
    assert isinstance(gate, ApprovedByFinance) and gate.min_total == 500

    with pytest.raises(TypeError, match="completion handler 'NotStorableGate'"):
        full_agent(tmp_path, completion_handlers=[NotStorableGate()]).serialize()
