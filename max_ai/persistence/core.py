"""
Run context persistence contract.

A ``RunContextStore`` saves and loads ``RunContext`` instances by
``run_id``. The framework provides reference implementations for
in-memory (tests) and filesystem (single-process default) storage;
production deployments are expected to implement the protocol against
their own backend (Postgres, Redis, S3, etc.).

Concurrency: the contract makes no guarantees beyond per-call
correctness. Implementations may be last-write-wins, may raise on
conflict, or may serialize via locks — the choice is per-backend.
Callers that need stronger semantics (optimistic concurrency,
distributed locks) layer those on top.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..types.run_context import RunContext


# Run IDs are used as filesystem paths and database keys. We restrict
# them to hex / alphanumeric / hyphen / underscore so no implementation
# has to defend against path traversal or SQL escaping.
_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_run_id(run_id: str) -> None:
    """Raise ``ValueError`` if ``run_id`` is unsafe to use as a key.

    Implementations should call this at the top of every method that
    accepts a ``run_id`` argument.
    """
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}: must match {_RUN_ID_PATTERN.pattern}"
        )


class RunContextStore(ABC):
    """Abstract contract for persisting ``RunContext`` across runs.

    The four methods cover the lifecycle:

    - ``save`` — write a context, overwriting any prior version with
      the same ``run_id``.
    - ``load`` — read a context, or return ``None`` if the key is
      unknown. Never raises on miss.
    - ``delete`` — remove a context. Idempotent — deleting an unknown
      key is not an error.
    - ``list`` — enumerate stored ``run_id`` values. Used for cleanup
      and debugging; not for application-level querying.

    All methods are async to accommodate I/O-bound backends. Pure
    in-memory implementations may simply ``await asyncio.sleep(0)``
    or return immediately.
    """

    @abstractmethod
    async def save(self, run_id: str, ctx: RunContext) -> None:
        """Persist a ``RunContext`` under ``run_id``.

        Overwrites any prior version. Implementations should make the
        write atomic when feasible — partial writes leave a corrupted
        state that ``load`` cannot recover from.

        Args:
            run_id: Identifier; must pass ``validate_run_id``.
            ctx: The context to persist. Implementations are expected
                to use Pydantic's ``model_dump`` / ``model_validate``
                for serialization.

        Raises:
            ValueError: If ``run_id`` is invalid.
        """
        ...

    @abstractmethod
    async def load(self, run_id: str) -> RunContext | None:
        """Load a previously saved ``RunContext``.

        Args:
            run_id: Identifier; must pass ``validate_run_id``.

        Returns:
            The stored ``RunContext``, or ``None`` if no entry exists
            for this ``run_id``.

        Raises:
            ValueError: If ``run_id`` is invalid.
        """
        ...

    @abstractmethod
    async def delete(self, run_id: str) -> None:
        """Delete a stored ``RunContext``.

        Idempotent — silently succeeds when the key is unknown.

        Args:
            run_id: Identifier; must pass ``validate_run_id``.

        Raises:
            ValueError: If ``run_id`` is invalid.
        """
        ...

    @abstractmethod
    async def list(self) -> list[str]:
        """Enumerate all stored ``run_id`` values.

        Order is implementation-defined (filesystem may return alpha,
        databases may return insertion order, etc.). Useful for
        debugging and bulk cleanup; not intended for application-level
        querying — push that into a real database.

        Returns:
            A list of ``run_id`` strings currently present in the store.
        """
        ...
