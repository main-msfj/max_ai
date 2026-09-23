"""Memory registry backends.

``SQLiteMemoryRegistry`` is lazy-loaded (PEP 562): it's stale against the
current ``base/memory.py`` contract (references a ``RecallQuery`` type that
no longer exists) and would break importing this package eagerly. Fix it
in sqlite.py, then this stays a plain re-export — no change needed here.
"""

import typing as t

from .local import LocalMemoryRegistry, LocalMemoryRegistryConfig
from .mongodb import MongoDBMemoryRegistry, MongoDBMemoryRegistryConfig

if t.TYPE_CHECKING:
    from .sqlite import SQLiteMemoryRegistry, SQLiteMemoryRegistryConfig

__all__ = [
    "LocalMemoryRegistry",
    "LocalMemoryRegistryConfig",
    "MongoDBMemoryRegistry",
    "MongoDBMemoryRegistryConfig",
    "SQLiteMemoryRegistry",
    "SQLiteMemoryRegistryConfig",
]


def __getattr__(name: str) -> t.Any:
    if name in {"SQLiteMemoryRegistry", "SQLiteMemoryRegistryConfig"}:
        from . import sqlite

        return getattr(sqlite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
