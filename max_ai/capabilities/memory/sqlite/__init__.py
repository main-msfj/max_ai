"""SQLite memory provider; the runtime is loaded only when requested."""

import typing as t

from ._model import SQLiteMemoryRegistryConfig

if t.TYPE_CHECKING:
    from ._registry import SQLiteMemoryRegistry

__all__ = ["SQLiteMemoryRegistry", "SQLiteMemoryRegistryConfig"]


def __getattr__(name: str) -> t.Any:
    if name == "SQLiteMemoryRegistry":
        from ._registry import SQLiteMemoryRegistry

        return SQLiteMemoryRegistry
    raise AttributeError(name)
