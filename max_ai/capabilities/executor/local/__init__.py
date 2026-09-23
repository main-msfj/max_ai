"""Local (same-process) executor, available through max_ai.capabilities.executor.local."""

from ._executor import LocalExecutor
from ._model import LocalExecutorConfig

__all__ = ["LocalExecutor", "LocalExecutorConfig"]
