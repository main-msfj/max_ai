"""Serializable config for OpenAIChatCompletionClient."""

import typing as t
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class OpenAIChatCompletionClientConfig(BaseModel):
    """Configuration for OpenAIChatCompletionClient serialization."""

    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
