"""Fixed executor infrastructure, shared by every backend (local/docker/modal).

Unlike the backends themselves (pluggable, under capabilities/executor/),
nothing here is meant to be swapped out — same rationale as core/tool/.
``worker.py`` isn't re-exported: it's a ``python -m`` entrypoint invoked by
name inside a sandboxed image, not a library import.
"""

from .process import run_process
from .reference import ToolReference, reference_for
from .remote import RemoteExecutor

__all__ = ["run_process", "ToolReference", "reference_for", "RemoteExecutor"]
