"""to_yaml / to_json write an agent compactly (provider + what differs from
the defaults), so it can be written by hand; from_yaml / from_json read it
back into the same agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai.client import OpenAIChatCompletionClient
from tests.agents.test_agent_serialization import full_agent


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-stored")


def stored(agent: Agent) -> dict:
    return agent.serialize().model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_the_compact_form_rebuilds_the_same_agent(tmp_path, fmt):
    agent = full_agent(tmp_path)
    text = getattr(agent, f"to_{fmt}")()
    assert stored(getattr(Agent, f"from_{fmt}")(text)) == stored(agent)
    assert "component_type" not in text and "plan_must_close: false" in agent.to_yaml()


def test_a_hand_written_yaml_needs_only_what_differs(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(f"""
provider: maxai.agents.Agent
config:
  name: helper
  description: Answers questions.
  instructions: |
    Be brief.
    Cite your sources.
  client:
    provider: maxai.llm.OpenAIChatCompletionClient
    config: {{model: gpt-4o-mini}}
  workspace:
    provider: max_ai.capabilities.workspace.local.LocalWorkspace
    config: {{root: {tmp_path}}}
""")
    agent = Agent.from_yaml(Path(path))
    assert (agent.name, agent.client.model) == ("helper", "gpt-4o-mini")
    assert agent.instructions == "Be brief.\nCite your sources.\n"
    assert "instructions: |" in agent.to_yaml()  # multi-line text stays readable


def test_full_keeps_everything(tmp_path):
    agent = Agent(name="a", description="d", instructions="i",
                  client=OpenAIChatCompletionClient(model="gpt-4o-mini"))
    assert agent.to_dict(full=True) == stored(agent)
    assert len(agent.to_yaml(full=True)) > 2 * len(agent.to_yaml())
