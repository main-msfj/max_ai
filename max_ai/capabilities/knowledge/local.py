"""
Filesystem-backed knowledge registry.

Each knowledge source is a single JSON file under
``{base_path}/knowledge/{name}.json``, holding a flat list of
``KnowledgeBlock`` entries. The agent uses ``search`` to pull relevant
blocks for a query.

Search uses local embedding cosine similarity — the same lightweight
embedding helper used by the SQLite memory/routine backends. Block
content is embedded at search time (these registries store no
precomputed vectors on disk).

The registry is read-only from the agent's perspective. Block ingestion
(chunking, vectorization, writing to disk) happens externally —
typically a separate pipeline that pre-populates the knowledge file.

Not concurrency-safe: simultaneous external writers can clobber each
other. Intended for local development and tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from pydantic import BaseModel

from ...base.embeddings import get_lightweight_embedding
from ...base.knowledge import CoreKnowledgeRegistry, KnowledgeToolMode
from ...core import KnowledgeBlock


class LocalKnowledgeRegistryConfig(BaseModel):
    name: str
    description: str
    base_path: str
    tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL

class LocalKnowledgeRegistry(CoreKnowledgeRegistry):
    component_schema = LocalKnowledgeRegistryConfig
    component_type = "knowledge"

    """Filesystem-backed implementation of ``CoreKnowledgeRegistry``.

    Layout::

        {base_path}/
            knowledge/
                {name}.json          # list[KnowledgeBlock]

    File shape::

        [
            {
                "content": "FastAPI is a modern web framework...",
                "score": null,
                "tokens": 14,
                "metadata": {"source": "docs.md"}
            },
            ...
        ]

    Each entry's ``score`` field on disk is ignored at search time —
    the registry computes a fresh score against the current query and
    returns blocks with that score attached. ``tokens`` and
    ``metadata`` are passed through unchanged.
    """

    def __init__(
        self,
        name: str,
        description: str,
        base_path: str | Path,
        tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL,
    ) -> None:
        super().__init__(
            name=name, description=description, tool_mode=tool_mode
        )
        self.base_path: Path = Path(base_path).expanduser().resolve()

    def _to_config(self) -> LocalKnowledgeRegistryConfig:
        return LocalKnowledgeRegistryConfig(
            name=self.name,
            description=self.description,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
        )

    @classmethod
    def _from_config(cls, config: LocalKnowledgeRegistryConfig) -> "LocalKnowledgeRegistry":
        return cls(
            name=config.name,
            description=config.description,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
        )

    # -------- PATH HELPERS -----------------------------------------------------------
    @property
    def _knowledge_dir(self) -> Path:
        return self.base_path / "knowledge"

    @property
    def _source_file(self) -> Path:
        return self._knowledge_dir / f"{self.name}.json"

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Ensure the knowledge directory exists. The source file
        itself is written externally — readers tolerate its absence."""
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        return None

    # -------- READ OPERATIONS -----------------------------------------------------------
    async def search(
        self, query: str, limit: int = 5
    ) -> list[KnowledgeBlock]:
        """Search this knowledge source for relevant blocks.

        Embeds the query and each block's ``content`` with the local
        lightweight embedding model and scores by cosine similarity.
        Blocks with non-positive similarity are dropped; the rest are
        sorted by score descending and the top ``limit`` are returned.

        Returns an empty list when the source file doesn't exist yet.
        """
        await self._ensure_connected()
        if not isinstance(query, str) or not query.strip():
            return []

        blocks = self._read_all()
        if not blocks:
            return []

        query_vector = get_lightweight_embedding(query)

        scored: list[tuple[float, KnowledgeBlock]] = []
        for block in blocks:
            block_vector = get_lightweight_embedding(block.content)
            score = self._cosine_similarity(query_vector, block_vector)
            if score <= 0:
                continue
            # Re-emit the block with the freshly computed score.
            scored.append(
                (
                    score,
                    KnowledgeBlock(
                        content=block.content,
                        score=score,
                        tokens=block.tokens,
                        metadata=block.metadata,
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [block for _, block in scored[:limit]]

    # -------- INTERNALS -----------------------------------------------------------
    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Cosine similarity between two embedding vectors.

        Returns 0.0 for empty, mismatched-length, or zero-norm vectors.
        """
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _read_all(self) -> list[KnowledgeBlock]:
        """Load every block in the source file.

        Returns ``[]`` when the file doesn't exist yet — first call
        for a new source is not an error. Corrupt file or invalid
        block fails loud.
        """
        path = self._source_file
        if not path.is_file():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Knowledge file {path} is not valid JSON: {e}"
            ) from e

        if not isinstance(raw, list):
            raise ValueError(
                f"Knowledge file {path} must contain a JSON array at the "
                f"top level, got {type(raw).__name__}"
            )

        result: list[KnowledgeBlock] = []
        for i, payload in enumerate(raw):
            try:
                result.append(KnowledgeBlock.model_validate(payload))
            except Exception as e:
                raise ValueError(
                    f"Knowledge file {path}: entry at index {i} is not a "
                    f"valid KnowledgeBlock: {e}"
                ) from e
        return result
