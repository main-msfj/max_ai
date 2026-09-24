"""Agent behaviour limits and portable constructor config."""

import typing as t

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """An Agent as storable data: ``Agent.serialize()`` builds it and
    ``Agent.deserialize()`` rebuilds the agent from it.

    Every component is a ``ComponentModel`` dict (provider + config). It holds
    no secrets (only env var names) and no code: tools written as Python
    functions cannot be stored, expose them through an MCP server instead.
    """

    name: str
    description: str
    instructions: str
    client: dict[str, t.Any]
    reasoning: dict[str, t.Any] | None = None
    workspace: dict[str, t.Any] | None = None
    executor: dict[str, t.Any] | None = None
    memory: dict[str, t.Any] | None = None
    skills: dict[str, t.Any] | None = None
    knowledge: list[dict[str, t.Any]] = Field(default_factory=list)
    compaction: dict[str, t.Any] | None = None
    toolset: list[dict[str, t.Any]] = Field(default_factory=list)
    mcp: list[dict[str, t.Any]] = Field(default_factory=list)
    gates: list[dict[str, t.Any]] = Field(
        default_factory=list, description="Completion gates: the framework's and your own.",
    )
    output_format: dict[str, t.Any] | None = Field(
        default=None, description="JSON Schema of the final answer.",
    )
    middlewares: list[dict[str, t.Any]] = Field(default_factory=list)
    prompt_layers: list[dict[str, t.Any]] = Field(
        default_factory=list,
        description="Custom prompt layers that replace matching defaults or append to the Agent stack.",
    )
