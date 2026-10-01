"""Retries for what is safe to repeat, done by the harness and never seen by the model.

Only infrastructure goes through here: calling the model, connecting to a
sandbox or an MCP server, syncing files. A command that already started is
never run twice (``pip install`` or sending an email is not idempotent): its
failure goes back to the model, which decides like with any other error.
"""

from __future__ import annotations

import asyncio
import random
import typing as t

from pydantic import BaseModel, ConfigDict, Field

T = t.TypeVar("T")


class RetryPolicy(BaseModel):
    """How many times to try, and how long to wait in between."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts: int = Field(default=4, ge=1, description="Tries in total: 1 call + retries.")
    base_delay: float = Field(default=1.0, ge=0, description="First wait, in seconds.")
    max_delay: float = Field(default=30.0, ge=0, description="Longest wait, in seconds.")

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after failed try ``attempt`` (0 is the first): what
        the provider asked for, else exponential with full jitter."""
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_delay)
        return random.uniform(0, min(self.max_delay, self.base_delay * 2**attempt))


def retry_after(error: BaseException) -> float | None:
    """The wait the provider asked for with this error, if any."""
    value = getattr(error, "retry_after", None)
    return float(value) if isinstance(value, (int, float)) else None


async def retrying(
    call: t.Callable[[], t.Awaitable[T]],
    *,
    policy: RetryPolicy,
    transient: t.Callable[[BaseException], bool],
    on_retry: t.Callable[[int, BaseException, float], t.Any] | None = None,
) -> T:
    """Await ``call()`` until it works, fails with a non-transient error, or
    ``policy.attempts`` run out (then the last error is raised).

    ``on_retry(attempt, error, delay)`` runs before each wait; ``attempt``
    counts from 1. Cancellation is never retried.
    """
    for attempt in range(policy.attempts):
        try:
            return await call()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if attempt + 1 >= policy.attempts or not transient(error):
                raise
            wait = policy.delay(attempt, retry_after(error))
            if on_retry is not None:
                on_retry(attempt + 1, error, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["RetryPolicy", "retry_after", "retrying"]
