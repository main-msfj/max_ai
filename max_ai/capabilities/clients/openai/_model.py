"""Serializable config for OpenAIChatCompletionClient."""

import typing as t
from typing import Literal

from pydantic import BaseModel, Field

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class OpenAIChatCompletionClientConfig(BaseModel):
    """Configuration for OpenAIChatCompletionClient serialization."""

    model: str
    # The env var name, never the key: the config may be stored anywhere.
    api_key_env: str | None = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
