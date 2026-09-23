"""Serializable config for OpenRouterChatCompletionClient."""

import typing as t

from pydantic import BaseModel, Field, SecretStr


class OpenRouterChatCompletionClientConfig(BaseModel):
    """Configuration for OpenRouterChatCompletionClient serialization."""

    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    reasoning: dict[str, t.Any] | None = None
    provider: dict[str, t.Any] | None = None
    app_name: str | None = None
    app_url: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
