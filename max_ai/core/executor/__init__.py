"""Fixed executor infrastructure, shared by every backend (local/docker/modal).

Unlike the backends themselves (pluggable, under capabilities/executor/),
nothing here is meant to be swapped out — same rationale as core/tool/.
"""

from .process import run_process
from .remote import RemoteExecutor

__all__ = ["run_process", "RemoteExecutor"]
