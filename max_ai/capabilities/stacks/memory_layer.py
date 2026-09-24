"""
Layer that injects persistent memory into the agent's system prompt.
"""

import typing as t
from pathlib import Path

from ...base.component import Component
from ...base.layer import CoreLayer
from ...core.model.stacks import StackConfig


class MemoryLayer(Component[StackConfig], CoreLayer):
    """
    Prompt layer that exposes persistent memory to the agent.

    Renders two pieces of context into the system prompt:
      1. The current snapshot of stored memories (``persistent_memories``)
         — already-known facts about the user, injected as background
         context the agent can use to personalize responses.
      2. The names of any memory CRUD tools available
         (``memory_tools``) — so the agent knows it can list, update,
         or delete memories during the turn. Tool descriptions live
         on each tool itself; the layer only advertises that they
         exist.

    If memory has no tools (e.g. ``MemoryToolMode.NONE``), the
    ``<memory_management>`` block disappears from the prompt — the
    agent treats memory as read-only background context.

    Template Variables:
        persistent_memories (required): List of ``MemoryRecord`` objects
            from ``CoreMemoryRegistry.get_context()``. Each block has
            ``category``, ``memory``, and ``updated``. Empty
            list is valid (no memories yet).
        memory_tools (optional): List of tool names exposed by the
            memory backend. Empty list / omitted means no CRUD
            available — the management block won't render.
    """

    component_schema = StackConfig
    component_type = "prompts"
    component_provider_override = "maxai.stacks.MemoryLayer"

    def __init__(
        self,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        """Initialize ``MemoryLayer``.

Parameters
----------
template : str | None
    Value supplied for ``template``.
load_from : str | Path | None
    Value supplied for ``load_from``.
extra_variables : dict[str, t.Any] | None
    Value supplied for ``extra_variables``."""
        super().__init__(
            name="MemoryLayer",
            template=template,
            load_from=load_from,
            extra_variables=extra_variables,
        )

    def _default_template(self) -> str:
        """Perform the internal ``default template`` operation for ``MemoryLayer``."""
        return self._load_file(self._DEFAULT_TEMPLATE_PATH / f"{self.name}.j2")

    def _required_variables(self) -> set[str]:
        """Perform the internal ``required variables`` operation for ``MemoryLayer``."""
        return {"persistent_memories"}

    def _optional_variables(self) -> set[str]:
        """Perform the internal ``optional variables`` operation for ``MemoryLayer``."""
        return {"memory_tools"}
