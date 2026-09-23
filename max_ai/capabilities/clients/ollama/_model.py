"""Serializable config for OllamaChatCompletionClient."""

import typing as t
from typing import Literal

from pydantic import BaseModel, Field

OllamaThinkingEffort = Literal["low", "medium", "high"]
OllamaThink = bool | OllamaThinkingEffort


class OllamaChatCompletionClientConfig(BaseModel):
    """Configuration for OllamaChatCompletionClient serialization."""

    model: str
    host: str
    # The env var name, never the key (local Ollama needs none).
    api_key_env: str | None = "OLLAMA_API_KEY"
    think: OllamaThink | None = None
    keep_alive: str | int | None = None
    options: dict[str, t.Any] = Field(default_factory=dict)
    config: dict[str, t.Any] = Field(default_factory=dict)
