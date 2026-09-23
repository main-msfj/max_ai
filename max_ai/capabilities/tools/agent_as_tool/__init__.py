"""Agent-as-tool provider with a separately importable config model."""

import typing as t

from ._model import AgentAsToolConfig

if t.TYPE_CHECKING:
    from ._tool import AgentAsTool

__all__ = ["AgentAsTool", "AgentAsToolConfig"]


def __getattr__(name: str) -> t.Any:
    if name == "AgentAsTool":
        from ._tool import AgentAsTool

        return AgentAsTool
    raise AttributeError(name)
