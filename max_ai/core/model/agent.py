"""Agent behaviour limits and portable constructor config."""

import typing as t

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Global configuration for agent behavior and execution limits."""

    summarize_tool_result: bool = Field(default=True)
    enable_self_reflection: bool = Field(default=False)
    max_loop_iterations: int = Field(default=20)
    tool_timeout: int = Field(default=300)
    tool_call_concurrency: int = Field(default=5)
    max_connection_retries: int = Field(default=3)
    exponential_backoff_base: float = Field(default=1.0)


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
    completion: dict[str, t.Any] = Field(
        default_factory=dict, description="RuntimeGateConfig options of the framework's gate.",
    )
    completion_handlers: list[dict[str, t.Any]] = Field(
        default_factory=list, description="The developer's own gates, as components.",
    )
    output_format: dict[str, t.Any] | None = Field(
        default=None, description="JSON Schema of the final answer.",
    )
    middlewares: list[dict[str, t.Any]] = Field(default_factory=list)
