"""Modal Sandbox executor, available through max_ai.capabilities.executor.modal.

Lazy (PEP 562): ``sync.py`` and ``modal_command.py`` are invoked inside the
sandbox as ``python -m max_ai.capabilities.executor.modal.sync`` /
``...modal_command`` — that always imports this package's __init__ first.
If it eagerly imported ``_executor`` (which imports ``.sync``), running
``sync.py`` as __main__ would double-import it, producing a
'found in sys.modules ... prior to execution' RuntimeWarning.
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
