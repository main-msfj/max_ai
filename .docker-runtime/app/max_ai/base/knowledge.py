"""
Core contract for agent external knowledge sources.

A knowledge source is a connection to an external retrieval backend
(vector DB, Notion, SharePoint, web search, etc.) that the agent can
query for information beyond its own context.

An agent may have several knowledge sources at once — each is an
independent backend with its own name. The LLM sees one search tool
per source, named ``search_{name}``, so it can pick the right one
based on the query.

A ``CoreKnowledgeRegistry`` exposes two surfaces to the agent:

  1. ``name``       → unique identifier that drives the tool's name.
  2. ``as_tool()``  → the search tool the LLM can call. Controlled by
                      ``tool_mode``.

Subclasses implement ``search`` (storage-level retrieval). The base
class handles tool generation.
"""

from __future__ import annotations

import logging
import re
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from .capability import CoreAgentCapabilities
from .tools import CoreTool
from ..tools.function_as_tool import FunctionAsTool
from ..types.tools import ToolApprovalMode
from ..core import KnowledgeBlock
from ..loggers import ScopedLogger

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope=["CoreKnowledgeRegistry"])


# Valid tool names for most providers: letters, digits, underscores, hyphens.
_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class KnowledgeToolMode(str, Enum):
    """Controls whether the knowledge search tool is exposed to the LLM.

    - ``NONE`` : No tool. The knowledge backend is still connectable and
                 usable by the framework (e.g. to pre-populate context),
                 but the LLM has no way to query it.
    - ``FULL`` : The ``search_{name}`` tool is exposed. The LLM can
                 query the backend on demand.
    """

    NONE = "none"
    FULL = "full"


class CoreKnowledgeRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base class for external knowledge sources."""

    def __init__(
        self,
        name: str,
        description: str,
        tool_mode: KnowledgeToolMode = KnowledgeToolMode.FULL,
    ) -> None:
        super().__init__()
        self.name: str = self._validate_name(name)
        self.description: str = self.require_type(
            description, str, "description"
        )
        self.tool_mode: KnowledgeToolMode = self.require_type(
            tool_mode, KnowledgeToolMode, "tool_mode"
        )

    @staticmethod
    def _validate_name(name: str) -> str:
        """Names become part of the tool identifier — keep them provider-safe."""
        if not isinstance(name, str) or not name:   # type: ignore
            raise TypeError("name must be a non-empty string")
        if not _VALID_NAME_RE.match(name):
            raise ValueError(
                f"Invalid knowledge name {name!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return name

    # -------- READ OPERATION -----------------------------------------------------------
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[KnowledgeBlock]:
        """Retrieve relevant blocks for a query.

        Returns up to ``limit`` blocks ranked by the backend's relevance
        criteria (vector similarity, BM25, hybrid, etc.). Each block
        contains content plus optional score and free-form metadata.
        """
        ...

    # -------- TOOL EXPOSURE -----------------------------------------------------------
    def as_tool(self) -> CoreTool | None:
        """
        Generate the search tool the LLM can call for this knowledge source.

        Returns ``None`` when ``tool_mode == NONE`` — the agent should
        simply skip it when collecting tools.

        The tool is named ``search_{self.name}``, so multiple sources
        in the same agent stay distinguishable.
        """
        if self.tool_mode == KnowledgeToolMode.NONE: # This doe sno make sense as knowledge must be injeccted other wise it will never be used
            return None
        return self._build_search_tool()

    @property
    def tools(self) -> list[CoreTool]:
        """Public capability surface for tool aggregation."""
        tool = self.as_tool()
        return [] if tool is None else [tool]

    # -------- TOOL FACTORIES -----------------------------------------------------------
    def _build_search_tool(self) -> CoreTool:
        """Tool: search this knowledge source."""

        async def search_knowledge(
            query: str, limit: int = 5
        ) -> list[dict[str, t.Any]]:
            """Search this knowledge source for relevant information.

            Returns blocks with content and metadata. Use this when the
            user's question needs information beyond your own knowledge —
            facts, documentation, or domain-specific content stored in
            this source.

            Args:
                query: Natural-language description of the information
                    needed. Keep it focused.
                limit: Maximum number of blocks to return.
            """
            await self._ensure_connected()
            blocks = await self.search(query, limit=limit)
            return [b.model_dump() for b in blocks]

        return FunctionAsTool(
            search_knowledge,
            name=f"search_{self.name}",
            description=self.description,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
