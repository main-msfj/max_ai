"""Core base for agent capability registries."""

from __future__ import annotations

import asyncio
import typing as t
from abc import ABC, abstractmethod

from .component import ComponentBase, ConfigT

if t.TYPE_CHECKING:
    from .tools import CoreTool


class CoreAgentCapabilities(ComponentBase[ConfigT], ABC):
    """Serializable registry with an async connection lifecycle.

    Capabilities are agent-facing resources such as memory, knowledge,
    routines, skills, and conversation context. They may expose tools,
    but they do not have to.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._connect_lock: asyncio.Lock = asyncio.Lock()

    @abstractmethod
    async def connect(self) -> None:
        """Open backend resources needed by this capability."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close backend resources opened by connect()."""
        ...

    async def _ensure_connected(self) -> None:
        """Connect once, safely, even under concurrent calls."""
        if not self._connected:
            async with self._connect_lock:
                if not self._connected:
                    await self.connect()
                    self._connected = True

    async def __aenter__(self) -> t.Self:
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc: t.Any) -> None:
        if self._connected:
            await self.disconnect()
            self._connected = False

    @property
    def tools(self) -> list["CoreTool"]:
        """Tools exposed by this capability, if any."""
        return []
