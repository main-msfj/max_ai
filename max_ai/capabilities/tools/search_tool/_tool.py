"""search_tools: find deferred tools and show how to call them."""

from __future__ import annotations

import logging
import typing as t

from ....base.embedding import CoreEmbedding
from ....base.tools import CoreTool, ToolContext
from ....core.embeddings import FastEmbedEmbedding, rank
from ....core.ranking import bm25, fuse, words
from ....core.termination.cancellation import CancellationToken
from ....core.tool.registry import ToolRegistry
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode

logger = logging.getLogger(__name__)


class SearchToolsTool(CoreTool):
    """Look up tools that are not loaded and return their schemas.

    Hybrid search: BM25 finds the exact words ("exa", "fetch"), embeddings
    find the meaning ("leer una página"), and RRF joins both rankings.
    Without an embedding model it searches by keywords only."""

    TOOL_NAME: t.ClassVar[str] = "search_tools"
    LIMIT: t.ClassVar[int] = 5
    MIN_SIMILARITY: t.ClassVar[float] = 0.3
    DESCRIPTION = (
        "Find tools that are not loaded yet, by name or by what they do, "
        "and return their descriptions and parameters."
    )
    WORDS: t.ClassVar[frozenset[str]] = frozenset(
        "a an and are as at be by for from in is it of on or the to with "
        "you your its this that their them can one".split()
    )

    def __init__(self, catalog: ToolRegistry) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
        )
        self._catalog = catalog
        self._embedding: CoreEmbedding | None = FastEmbedEmbedding()

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact tool names, to load several at once.",
                },
                "query": {
                    "type": "string",
                    "description": "What the tool should do, in a few words.",
                },
            },
        }

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        names = [str(name) for name in tool_request.parameters.get("names") or []] # type: ignore 
        query = str(tool_request.parameters.get("query") or "").strip()
        if not names and not query:
            return ToolResult.execution_error(tool_request.id, "Pass names, a query, or both.")

        found = [tool for name in names if (tool := self._catalog.get(name)) is not None]
        unknown = [name for name in names if self._catalog.get(name) is None]
        for tool in await self.search(query) if query else []:
            if tool not in found:
                found.append(tool)

        payload: dict[str, t.Any] = {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in found
            ]
        }
        if unknown or not found:
            available = ", ".join(tool.name for tool in self._catalog.deferred_tools()) or "none"
            missed = f"No tool named {', '.join(unknown)}. " if unknown else ""
            payload["hint"] = f"{missed}Tools you can load: {available}."
        return ToolResult.success_result(tool_request.id, payload)

    async def search(self, query: str) -> list[CoreTool]:
        """The best ``LIMIT`` deferred tools for ``query``, best first."""
        tools = self._catalog.deferred_tools()
        if not tools:
            return []

        # Applying raking
        rankings = [self._by_keywords(query, tools), await self._by_meaning(query, tools)]
        by_name = {tool.name: tool for tool in tools}
        return [by_name[name] for name in fuse(rankings)[: self.LIMIT]]

    def _by_keywords(self, query: str, tools: list[CoreTool]) -> list[str]:
        # The name counts twice: it is the strongest hint of what a tool does.
        docs = [words(t.name, self.WORDS) * 2 + words(t.description, self.WORDS) for t in tools]
        scores = bm25(words(query, self.WORDS), docs)
        scored = sorted(zip(scores, (tool.name for tool in tools)), key=lambda pair: -pair[0])
        return [name for score, name in scored if score > 0]

    async def _by_meaning(self, query: str, tools: list[CoreTool]) -> list[str]:
        if self._embedding is None:
            return []

        # Extracting Tool Info
        texts = [f"{tool.name.replace('_', ' ')}: {tool.description}" for tool in tools]
        try:
            vectors = await self._embedding.embed([query, *texts])
        except Exception:  # not installed or no model download: keywords still work
            logger.warning("search_tools: embeddings unavailable", exc_info=True)
            self._embedding = None
            return []

        best = rank(vectors[0], tools, vectors[1:], limit=self.LIMIT, min_score=self.MIN_SIMILARITY)
        return [tool.name for _, tool in best]
