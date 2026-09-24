"""Contract for counting what each user consumes, per period.

``BudgetMiddleware`` reads a user's usage when a task starts and adds to it
after every model call. The store only counts; the limits live in
``QuotaLimits``. Backends must make ``add`` atomic: two runs of the same
user may charge at the same time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..core.model.quota import QuotaUsage
from .component import CoreLifecycleComponent
from .session_store import validate_id


class CoreQuotaStore(CoreLifecycleComponent[BaseModel], ABC):
    """
    Define the persistent interface for per-user quota usage.
    """
    component_type = "quota_store"

    async def usage(self, user_id: str, period_key: str) -> QuotaUsage:
        """What ``user_id`` used in that period (zero if nothing yet)."""
        validate_id("user_id", user_id)
        await self._ensure_connected()
        return await self._usage(user_id, period_key)

    async def add(self, user_id: str, period_key: str, delta: QuotaUsage) -> QuotaUsage:
        """Add ``delta`` to the period and return the new total."""
        validate_id("user_id", user_id)
        await self._ensure_connected()
        return await self._add(user_id, period_key, delta)

    # -------- BACKEND HOOKS -----------------------------------------------------------
    @abstractmethod
    async def _usage(self, user_id: str, period_key: str) -> QuotaUsage:
        """
        Read the current usage for one user and period.

        Parameters
        ----------
        user_id : str
            Identifier for the user scope.
        period_key : str
            Key that identifies the quota period.

        Returns
        -------
        QuotaUsage
            The usage total after the operation.
        """
        ...

    @abstractmethod
    async def _add(self, user_id: str, period_key: str, delta: QuotaUsage) -> QuotaUsage:
        """
        Atomically add usage to one user and period.

        Parameters
        ----------
        user_id : str
            Identifier for the user scope.
        period_key : str
            Key that identifies the quota period.
        delta : QuotaUsage
            Usage to add to the stored total.

        Returns
        -------
        QuotaUsage
            The usage total after the operation.
        """
        ...


__all__ = ["CoreQuotaStore"]
