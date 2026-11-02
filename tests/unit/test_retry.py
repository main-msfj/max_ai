"""core/retry.py: what is retried, how many times, how long it waits."""

from __future__ import annotations

import asyncio

import pytest

from max_ai.core.retry import RetryPolicy, retrying


def test_delay_follows_retry_after_capped_else_jittered_backoff():
    policy = RetryPolicy(base_delay=1, max_delay=10)
    assert policy.delay(0, retry_after=4) == 4
    assert policy.delay(0, retry_after=60) == 10
    assert all(0 <= policy.delay(3) <= 8 for _ in range(50))
    assert all(policy.delay(9) <= 10 for _ in range(50))


async def test_retrying_stops_at_success_or_a_permanent_error():
    calls: list[int] = []
    seen: list[tuple[int, float]] = []
    policy = RetryPolicy(attempts=4, base_delay=0)

    async def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"

    result = await retrying(flaky, policy=policy, transient=lambda e: isinstance(e, ConnectionError),
                            on_retry=lambda attempt, error, delay: seen.append((attempt, delay)))
    assert result == "ok" and len(calls) == 3
    assert [attempt for attempt, _ in seen] == [1, 2]

    async def broken() -> None:
        calls.append(1)
        raise ValueError("bad input")

    calls.clear()
    with pytest.raises(ValueError):
        await retrying(broken, policy=policy, transient=lambda e: isinstance(e, ConnectionError))
    assert len(calls) == 1


async def test_retrying_gives_up_after_its_attempts():
    calls: list[int] = []

    async def down() -> None:
        calls.append(1)
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await retrying(down, policy=RetryPolicy(attempts=3, base_delay=0), transient=lambda e: True)
    assert len(calls) == 3


async def test_cancellation_is_never_retried():
    calls: list[int] = []

    async def cancelled() -> None:
        calls.append(1)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retrying(cancelled, policy=RetryPolicy(base_delay=0), transient=lambda e: True)
    assert len(calls) == 1
