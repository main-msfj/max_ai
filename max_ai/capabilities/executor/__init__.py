"""Execution providers for the new Agent; old executors remain in executor/."""

from ...base.executor import ExecutionSession, ExecutorBase

__all__ = [
    "ExecutorBase", "ExecutionSession",
    "LocalExecutor", "LocalExecutorConfig",
    "DockerExecutor", "DockerExecutorConfig",
    "ModalExecutor", "ModalExecutorConfig",
]


def __getattr__(name):
    """Resolve a lazily exported attribute.

Parameters
----------
name
    Value supplied for ``name``."""
    if name == "LocalExecutor":
        from .local import LocalExecutor
        return LocalExecutor
    if name == "LocalExecutorConfig":
        from .local import LocalExecutorConfig
        return LocalExecutorConfig
    if name == "DockerExecutor":
        from .docker import DockerExecutor
        return DockerExecutor
    if name == "DockerExecutorConfig":
        from .docker import DockerExecutorConfig
        return DockerExecutorConfig
    if name == "ModalExecutor":
        from .modal import ModalExecutor
        return ModalExecutor
    if name == "ModalExecutorConfig":
        from .modal import ModalExecutorConfig
        return ModalExecutorConfig
    raise AttributeError(name)
