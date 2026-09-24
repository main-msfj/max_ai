"""Embeddings from the OpenAI API (or any OpenAI-compatible endpoint)."""

from __future__ import annotations

import os
import typing as t

from pydantic import BaseModel, Field

from ...base.embedding import CoreEmbedding


class OpenAIEmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Env var name, never the key.")
    base_url: str | None = None
    dimensions: int | None = Field(default=None, gt=0)


class OpenAIEmbedding(CoreEmbedding):
    """No model to download: fits serverless. The key is read from
    ``api_key_env`` when the first text is embedded."""

    component_schema = OpenAIEmbeddingConfig
    component_provider_override = "max_ai.core.embeddings.OpenAIEmbedding"
    batch_size = 256

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.config = OpenAIEmbeddingConfig(
            model=model, api_key_env=api_key_env, base_url=base_url, dimensions=dimensions,
        )
        self._client: t.Any = None

    @property
    def model_id(self) -> str:
        return f"openai:{self.config.model}:{self.config.dimensions or 'default'}"

    def _to_config(self) -> OpenAIEmbeddingConfig:
        return self.config.model_copy()

    @classmethod
    def _from_config(cls, config: OpenAIEmbeddingConfig) -> OpenAIEmbedding:
        return cls(config.model, api_key_env=config.api_key_env,
                   base_url=config.base_url, dimensions=config.dimensions)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            from openai import AsyncOpenAI

            key = os.getenv(self.config.api_key_env)
            if not key:
                raise ValueError(f"Set {self.config.api_key_env} to use OpenAIEmbedding")
            self._client = AsyncOpenAI(api_key=key, base_url=self.config.base_url)
        options: dict[str, t.Any] = {"model": self.config.model, "input": texts}
        if self.config.dimensions:
            options["dimensions"] = self.config.dimensions
        response = await self._client.embeddings.create(**options)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
