"""Serializable configuration for AgentAsTool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentAsToolConfig(BaseModel):
    agent: dict[str, Any]
    input_name: str = "task"
    strategy: str = "last"
