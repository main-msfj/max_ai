"""search_tools: find deferred tools and show how to call them."""

from __future__ import annotations

import logging
import typing as t

from ....base.embedding import CoreEmbedding
from ....base.tools import CoreTool, ToolContext
from ....core.embeddings import rank
from ....core.retrieval import bm25, fuse, words
from ....core.tool.registry import ToolRegistry
from ....types.tool_call import ToolCallRecord, ToolResult
from ....types.tools import ToolApprovalMode

logger = logging.getLogger(__name__)


class SearchToolsTool(CoreTool):
    """Look up tools that are not loaded and return their schemas.

    It searches by keywords (BM25), which finds the exact words ("exa",
    "fetch"). Give it an ``embedding`` to also search by meaning ("leer una
    página"); RRF then joins both rankings."""

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

    def __init__(self, catalog: ToolRegistry, embedding: CoreEmbedding | None = None) -> None:
        super().__init__(
            name=self.TOOL_NAME,
            description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
        )
        self._catalog = catalog
        self._embedding = embedding
        self._vectors: dict[str, t.Sequence[float]] = {}  # a tool's text -> its vector

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

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        given: list[t.Any] = tool_request.parameters.get("names") or []
        names = [str(name) for name in given]
        query = str(tool_request.parameters.get("query") or "").strip()
        if not names and not query:
            return ToolResult.execution_error(tool_request.id, "Pass names, a query, or both.")

        # 1. By name: only deferred tools are loaded; the rest is told apart below.
        named = {name: self._catalog.get(name) for name in names}
        found = [tool for tool in named.values() if tool is not None and tool.deferred]
        loaded = [name for name, tool in named.items() if tool is not None and not tool.deferred]
        unknown = [name for name, tool in named.items() if tool is None]

        # 2. By what they do.
        for tool in await self.search(query) if query else []:
            if tool not in found:
                found.append(tool)
        payload: dict[str, t.Any] = {
            "tools": [
                {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
                for tool in found
            ]
        }

        # 3. Say what went wrong with a name, and what can be loaded.
        hints: list[str] = []
        if loaded:
            hints.append(f"Already loaded, call directly: {', '.join(loaded)}.")
        if unknown:
            hints.append(f"No tool named {', '.join(unknown)}.")
        if unknown or not found:
            available = ", ".join(tool.name for tool in self._catalog.deferred_tools()) or "none"
            hints.append(f"Tools you can load: {available}.")
        if hints:
            payload["hint"] = " ".join(hints)
        return ToolResult.success_result(tool_request.id, payload)

    async def search(self, query: str) -> list[CoreTool]:
        """The best ``LIMIT`` deferred tools for ``query``, best first."""
        tools = self._catalog.deferred_tools()
        if not tools:
            return []

        # Keywords always; meaning only with an embedding. RRF joins the rankings.
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

        # 1. Embed the query and only the tools not seen before.
        texts = [f"{tool.name.replace('_', ' ')}: {tool.description}" for tool in tools]
        new = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        try:
            vectors = await self._embedding.embed([query, *new])
        except Exception:  # not installed or no model download: keywords still work
            logger.warning("search_tools: embeddings unavailable", exc_info=True)
            self._embedding = None
            return []
        self._vectors.update(zip(new, vectors[1:]))

        # 2. The closest tools to the query.
        known = [self._vectors[text] for text in texts]
        best = rank(vectors[0], tools, known, limit=self.LIMIT, min_score=self.MIN_SIMILARITY)
        return [tool.name for _, tool in best]
