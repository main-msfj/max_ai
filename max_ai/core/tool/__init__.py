"""Tool runtime: record state, registry and dispatcher.

Dispatcher and registry load lazily (PEP 562): ``types.run_context`` imports
``core.tool.state``, and loading the dispatcher here eagerly would import
``RunContext`` back while it is still initializing.
"""

import typing as t

from .state import ToolState

if t.TYPE_CHECKING:
    from .dispatcher import ToolDispatcher
    from .registry import ToolRegistry

__all__ = ["ToolDispatcher", "ToolRegistry", "ToolState"]


def __getattr__(name: str) -> t.Any:
    if name == "ToolDispatcher":
        from .dispatcher import ToolDispatcher

        return ToolDispatcher
    if name == "ToolRegistry":
        from .registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
