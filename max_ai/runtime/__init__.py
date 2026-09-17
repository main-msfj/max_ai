"""Execution providers for the new Agent; old executors remain in executor/."""

from ..base.runtime_executor import Executor, ExecutionSession

__all__ = ["Executor", "ExecutionSession", "LocalExecutor", "DockerExecutor", "ModalExecutor"]


def __getattr__(name):
    if name == "LocalExecutor":
        from .local import LocalExecutor
        return LocalExecutor
    if name == "DockerExecutor":
        from .docker import DockerExecutor
        return DockerExecutor
    if name == "ModalExecutor":
        from .modal import ModalExecutor
        return ModalExecutor
    raise AttributeError(name)
