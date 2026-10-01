"""Modal Sandbox executor, available through max_ai.capabilities.executor.modal.

Lazy (PEP 562): ``ModalExecutor`` is imported on first use, so importing
this package never needs the optional Modal SDK.
"""

import typing as t

if t.TYPE_CHECKING:
    from ._executor import ModalExecutor

from ._model import PACKAGE_REGISTRIES, ModalExecutorConfig

__all__ = ["ModalExecutor", "ModalExecutorConfig", "PACKAGE_REGISTRIES"]


def __getattr__(name: str) -> t.Any:
    """Resolve a lazily exported attribute.

Parameters
----------
name : str
    Value supplied for ``name``."""
    if name == "ModalExecutor":
        from ._executor import ModalExecutor

        return ModalExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
