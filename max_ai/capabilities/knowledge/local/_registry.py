"""
Filesystem-backed knowledge registry.

Each knowledge source is a single JSON file under
``{base_path}/knowledge/{name}.json``, holding a flat list of
``KnowledgeBlock`` entries. The agent uses ``search`` to pull relevant
blocks for a query.

Search uses cosine similarity of this registry's ``embedding`` (default:
the local multilingual ``FastEmbedEmbedding``). Blocks are embedded at
search time in one batch; the embedding caches their vectors, so only new
or changed blocks are embedded again (nothing is stored on disk).

The registry is read-only from the agent's perspective. Block ingestion
(chunking, vectorization, writing to disk) happens externally —
typically a separate pipeline that pre-populates the knowledge file.

Not concurrency-safe: simultaneous external writers can clobber each
other. Intended for local development and tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from ....base.embedding import CoreEmbedding
from ....base.knowledge import CoreKnowledgeRegistry, KnowledgeToolMode
from ....core import KnowledgeBlock
from ....core.embeddings import FastEmbedEmbedding, rank
from ._model import LocalKnowledgeRegistryConfig


class LocalKnowledgeRegistry(CoreKnowledgeRegistry):
    """LocalKnowledgeRegistry provides the LocalKnowledgeregistry implementation."""
    component_provider_override = "max_ai.capabilities.knowledge.local.LocalKnowledgeRegistry"
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
                "tokens": 14,
                "metadata": {"source": "docs.md"}
            },
            ...
        ]

    Similarity score is computed against the current query purely to
    rank and filter blocks (see ``search``) — it isn't attached to the
    returned blocks or stored on disk, since a raw cosine value means
    nothing to the model on its own. ``tokens`` and ``metadata`` are
    passed through unchanged.
    """

    def __init__(
        self,
        name: str,
        description: str,
        base_path: str | Path,
        tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL,
        embedding: CoreEmbedding | None = None,
    ) -> None:
        """Initialize ``LocalKnowledgeRegistry``.

Parameters
----------
name : str
    Value supplied for ``name``.
description : str
    Value supplied for ``description``.
base_path : str | Path
    Value supplied for ``base_path``.
tool_mode : KnowledgeToolMode
    Value supplied for ``tool_mode``.
embedding : CoreEmbedding | None
    Value supplied for ``embedding``."""
        super().__init__(
            name=name, description=description, tool_mode=tool_mode
        )
        self.base_path: Path = Path(base_path).expanduser().resolve()
        # Knowledge search is meaning-only (no word-search fallback), so it
        # always needs a real embedding; default to the local multilingual
        # model (``maxai[embeddings]``) rather than requiring one every time.
        self.embedding = embedding if embedding is not None else FastEmbedEmbedding()

    def _to_config(self) -> LocalKnowledgeRegistryConfig:
        """Build the serializable configuration for ``LocalKnowledgeRegistry``."""
        return LocalKnowledgeRegistryConfig(
            name=self.name,
            description=self.description,
            base_path=str(self.base_path),
            tool_mode=self.tool_mode,
            embedding=self.embedding.serialize().model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(cls, config: LocalKnowledgeRegistryConfig) -> "LocalKnowledgeRegistry":
        """Create an instance from its configuration for ``LocalKnowledgeRegistry``.

Parameters
----------
config : LocalKnowledgeRegistryConfig
    Value supplied for ``config``."""
        return cls(
            name=config.name,
            description=config.description,
            base_path=config.base_path,
            tool_mode=config.tool_mode,
            embedding=CoreEmbedding.deserialize(config.embedding) if config.embedding else None,
        )

    # -------- PATH HELPERS -----------------------------------------------------------
    @property
    def _knowledge_dir(self) -> Path:
        """Perform the internal ``knowledge dir`` operation for ``LocalKnowledgeRegistry``."""
        return self.base_path / "knowledge"

    @property
    def _source_file(self) -> Path:
        """Perform the internal ``source file`` operation for ``LocalKnowledgeRegistry``."""
        return self._knowledge_dir / f"{self.name}.json"

    # -------- LIFECYCLE -----------------------------------------------------------
    async def connect(self) -> None:
        """Ensure the knowledge directory exists. The source file
        itself is written externally — readers tolerate its absence."""
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        """Release resources held for ``LocalKnowledgeRegistry``."""
        return None

    # -------- READ OPERATIONS -----------------------------------------------------------
    async def search(
        self, query: str, limit: int = 5
    ) -> list[KnowledgeBlock]:
        """Search this knowledge source for relevant blocks.

        Embeds the query and each block's ``content`` with this
        registry's embedding and scores by cosine similarity,
        purely to rank/filter — the score itself is never returned or
        stored (see the class docstring). Blocks with non-positive
        similarity are dropped; the rest are sorted by score
        descending and the top ``limit`` are returned.

        Returns an empty list when the source file doesn't exist yet.
        """
        await self._ensure_connected()
        if not isinstance(query, str) or not query.strip():
            return []

        blocks = self._read_all()
        if not blocks:
            return []

        # One batch; blocks already embedded come from the embedding's cache.
        query_vector, *vectors = await self.embedding.embed([query, *(b.content for b in blocks)])
        return [block for _, block in rank(query_vector, blocks, vectors, limit=limit)]

    # -------- INTERNALS -----------------------------------------------------------
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
