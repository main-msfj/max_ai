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
    async def _usage(self, user_id: str, period_key: str) -> QuotaUsage: ...

    @abstractmethod
    async def _add(self, user_id: str, period_key: str, delta: QuotaUsage) -> QuotaUsage: ...


__all__ = ["CoreQuotaStore"]
