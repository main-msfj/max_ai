"""Serializable config for OpenRouterChatCompletionClient."""

import typing as t

from pydantic import BaseModel, Field


class OpenRouterChatCompletionClientConfig(BaseModel):
    """Configuration for OpenRouterChatCompletionClient serialization."""

    model: str
    # The env var name, never the key: the config may be stored anywhere.
    api_key_env: str | None = "OPENROUTER_API_KEY"
    base_url: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    reasoning: dict[str, t.Any] | None = None
    provider: dict[str, t.Any] | None = None
    app_name: str | None = None
    app_url: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
