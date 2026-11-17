"""What is stored of an OpenRouter client. Never the key."""

import typing as t

from pydantic import BaseModel, Field

from ....core.model.llm import ModelConfig
from ....core.retry import RetryPolicy


class OpenRouterChatCompletionClientConfig(BaseModel):
    model: str
    api_key_env: str | None = "OPENROUTER_API_KEY"  # the variable's name, never the key
    base_url: str | None = None
    fallback_models: list[str] = Field(default_factory=list[str])
    provider: dict[str, t.Any] | None = None
    app_name: str | None = None
    app_url: str | None = None
    options: dict[str, t.Any] = Field(default_factory=dict[str, t.Any])  # sent as they are
    config: ModelConfig | None = None
    retry: RetryPolicy | None = None
