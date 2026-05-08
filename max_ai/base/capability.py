"""Core base for agent capability registries."""

from __future__ import annotations

import typing as t
from abc import ABC

from .component import ConfigT, CoreLifecycleComponent

if t.TYPE_CHECKING:
    from .tools import CoreTool


class CoreAgentCapabilities(CoreLifecycleComponent[ConfigT], ABC):
    """Serializable registry with an async connection lifecycle.

    Capabilities are agent-facing resources such as memory, knowledge,
    routines, skills, and conversation context. They may expose tools,
    but they do not have to.
    """

    @property
    def tools(self) -> list["CoreTool"]:
        """Tools exposed by this capability, if any."""
        return []
