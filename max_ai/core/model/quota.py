"""A user's quota: limits per period and what has been used."""

from __future__ import annotations

import typing as t
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, model_validator


class QuotaUsage(BaseModel):
    """What a user used in one period."""

    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    tasks: int = Field(default=0, ge=0)

    def __add__(self, other: QuotaUsage) -> QuotaUsage:
        return QuotaUsage(
            tokens=self.tokens + other.tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            tasks=self.tasks + other.tasks,
        )


class QuotaLimits(BaseModel):
    """Per-user limits that reset every ``period`` (UTC); ``None`` = no limit."""

    period: t.Literal["day", "month"] = "day"
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_tasks: int | None = Field(default=None, gt=0, description="New inputs per period.")

    @model_validator(mode="after")
    def _some_limit(self) -> QuotaLimits:
        if self.max_tokens is None and self.max_cost_usd is None and self.max_tasks is None:
            raise ValueError("QuotaLimits needs at least one limit")
        return self

    def period_key(self, now: datetime) -> str:
        """Storage key of the period ``now`` falls in, e.g. ``day:2026-09-23``."""
        return f"day:{now:%Y-%m-%d}" if self.period == "day" else f"month:{now:%Y-%m}"

    def resets_at(self, now: datetime) -> datetime:
        start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if self.period == "day":
            return start + timedelta(days=1)
        first = start.replace(day=1)
        return first.replace(year=first.year + 1, month=1) if first.month == 12 else first.replace(month=first.month + 1)

    def exceeded(self, used: QuotaUsage, now: datetime, *, starting_task: bool = False) -> str | None:
        """Why the quota is used up, said for the user; ``None`` if it isn't.
        The task limit only applies when a new task wants to start."""
        checks = [
            (self.max_tokens, used.tokens, "{limit:,} tokens"),
            (self.max_cost_usd, used.cost_usd, "${limit:.2f}"),
        ]
        if starting_task:
            checks.append((self.max_tasks, used.tasks, "{limit:,} tasks"))
        for limit, value, text in checks:
            if limit is not None and value >= limit:
                period = "daily" if self.period == "day" else "monthly"
                return (
                    f"Your {period} quota ({text.format(limit=limit)}) is used up; "
                    f"it resets at {self.resets_at(now):%Y-%m-%d %H:%M} UTC."
                )
        return None


__all__ = ["QuotaLimits", "QuotaUsage"]
