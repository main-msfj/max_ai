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


class AgentComponentConfig(BaseModel):
    """Portable constructor configuration for an ``Agent``."""

    name: str
    description: str
    instructions: str
    client: dict[str, t.Any]
    config: AgentConfig = Field(default_factory=AgentConfig)
    toolset: list[dict[str, t.Any]] = Field(default_factory=list)
    capabilities: list[dict[str, t.Any]] = Field(default_factory=list)
    workspace: dict[str, t.Any] | None = None
    middlewares: list[dict[str, t.Any]] = Field(default_factory=list)
    framework_layers: list[dict[str, t.Any]] = Field(default_factory=list)
    executor: dict[str, t.Any] | None = None
    output_format: str | None = None
    priority_tools: list[str] = Field(default_factory=list)
