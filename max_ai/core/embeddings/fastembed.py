"""The default embedding: a small local model (downloaded once, ~90MB)."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from ...base.embedding import CoreEmbedding
from .lightweight import DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL, _get_text_embedding_model


class FastEmbedEmbeddingConfig(BaseModel):
    model_name: str = DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL


class FastEmbedEmbedding(CoreEmbedding):
    """Runs on CPU in this process; needs ``pip install 'maxai[embeddings]'``.
    Good for local and server deployments; for serverless, where downloading
    a model per cold start is slow, prefer ``OpenAIEmbedding``."""

    component_schema = FastEmbedEmbeddingConfig
    component_provider_override = "max_ai.core.embeddings.FastEmbedEmbedding"

    def __init__(self, model_name: str = DEFAULT_LIGHTWEIGHT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name

    @property
    def model_id(self) -> str:
        return f"fastembed:{self.model_name}"

    def _to_config(self) -> FastEmbedEmbeddingConfig:
        return FastEmbedEmbeddingConfig(model_name=self.model_name)

    @classmethod
    def _from_config(cls, config: FastEmbedEmbeddingConfig) -> FastEmbedEmbedding:
        return cls(model_name=config.model_name)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        # CPU-bound: keep the event loop free for other runs.
        def run() -> list[list[float]]:
            model = _get_text_embedding_model(self.model_name)
            return [list(map(float, vector)) for vector in model.embed(texts)]

        return await asyncio.to_thread(run)
