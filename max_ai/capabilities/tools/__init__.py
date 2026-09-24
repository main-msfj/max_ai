"""Native and helper tools.

All re-exports are lazy (resolved on first attribute access). This keeps
``import max_ai.tools.plan`` — a pure data model needed by ``core`` and
``types`` — free of the heavier tool modules, which import ``base``/
``core`` themselves and would otherwise create an import cycle.
"""

import importlib
import typing as t

__all__ = [
    "CoreTool",
    "CoreRuntimeTool",
    "tool",
    "FunctionAsTool",
    "BashTool",
    "FileSystemTools",
    "FileSystem",
    "UserFileSystem",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CoreTool": ("max_ai.base.tools", "CoreTool"),
    "CoreRuntimeTool": ("max_ai.base.tools", "CoreRuntimeTool"),
    "tool": ("max_ai.capabilities.tools.decorator", "tool"),
    "FunctionAsTool": ("max_ai.capabilities.tools.function_as_tool", "FunctionAsTool"),
    "BashTool": ("max_ai.capabilities.tools.bash", "BashTool"),
    "FileSystem": ("max_ai.capabilities.tools.file_system", "FileSystem"),
    "FileSystemTools": ("max_ai.capabilities.tools.file_system", "FileSystemTools"),
    "UserFileSystem": ("max_ai.capabilities.tools.file_system", "UserFileSystem"),
}

if t.TYPE_CHECKING:  # static analyzers see the real symbols
    from ...base.tools import CoreRuntimeTool, CoreTool
    from .bash import BashTool
    from .decorator import tool
    from .file_system import FileSystem, FileSystemTools, UserFileSystem
    from .function_as_tool import FunctionAsTool


def __getattr__(name: str) -> t.Any:
    """Resolve a lazily exported attribute.

Parameters
----------
name : str
    Value supplied for ``name``."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr = target
    return getattr(importlib.import_module(module_name), attr)
