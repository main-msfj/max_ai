"""Budget for one task and quota per user: tokens, cost, time and calls.

Per task (``max_*``): a task is what one new input starts; the turn that
goes on after an approval or a question keeps counting. Time counts only while
the agent runs, never while it waits for the user.

Per user (``quota`` + ``quota_store``): limits per day or month shared by
all the user's tasks, counted in a ``CoreQuotaStore`` so every process and
every serverless invocation sees the same usage. It is charged after each
model call, so concurrent tasks of one user may go slightly over: the
next check stops them.

When a limit is reached:
- the next model call does not happen: the turn ends with
  ``finish_reason="budget_exceeded"`` and a message for the user;
- tools the model already asked for do not run: they answer
  "not run, budget used up", so the transcript stays consistent.
"""

from __future__ import annotations

import time
import typing as t
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from pydantic import Field, model_validator

from ...base.middleware import (
    CoreMiddleware,
    MiddlewareConfig,
    ModelCall,
    NextModel,
    NextStream,
    NextTool,
    NextTurn,
)
from ...base.quota_store import CoreQuotaStore
from ...core.model.quota import QuotaLimits, QuotaUsage
from ...types.completions import LLMChunk, LLMResponse, Usage
from ...types.tool_call import ToolCallRecord, ToolCallStatus, ToolResult

if t.TYPE_CHECKING:
    from pydantic import JsonValue

    from ...core.event_type import CoreEvent
    from ...core.model.llm import ModelConfig
    from ...types.run_context import RunContext

EXCEEDED = "budget_exceeded"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BudgetConfig(MiddlewareConfig):
    """Limits per task and quota per user; ``None`` = no limit."""

    max_tokens: int | None = Field(default=None, gt=0, description="Input + output tokens.")
    max_cost_usd: float | None = Field(
        default=None, gt=0, description="Needs the model's prices in its ModelConfig.",
    )
    max_seconds: float | None = Field(default=None, gt=0, description="Active time, not waiting.")
    max_model_calls: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    quota: QuotaLimits | None = None
    quota_store: dict[str, t.Any] | None = Field(default=None, description="A CoreQuotaStore.")

    @model_validator(mode="after")
    def _consistent(self) -> BudgetConfig:
        if (self.quota is None) != (self.quota_store is None):
            raise ValueError("quota and quota_store go together")
        limits = self.model_dump(exclude={"quota", "quota_store"}).values()
        if self.quota is None and all(value is None for value in limits):
            raise ValueError("BudgetMiddleware needs at least one limit")
        return self


class BudgetMiddleware(CoreMiddleware):
    """Stop a task before it spends more than it, or its user, may."""

    component_provider_override = "max_ai.capabilities.middleware.BudgetMiddleware"
    component_schema = BudgetConfig
    stop_message = "Budget exceeded"

    def __init__(
        self,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        max_seconds: float | None = None,
        max_model_calls: int | None = None,
        max_tool_calls: int | None = None,
        *,
        quota: QuotaLimits | None = None,
        quota_store: CoreQuotaStore | None = None,
    ) -> None:
        # Validates the limits once, at construction.
        BudgetConfig(
            max_tokens=max_tokens, max_cost_usd=max_cost_usd, max_seconds=max_seconds,
            max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
            quota=quota, quota_store={} if quota_store is not None else None,
        )
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_seconds = max_seconds
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.quota = quota
        self.quota_store = quota_store

    def _to_config(self) -> BudgetConfig:
        store = self.quota_store.serialize().model_dump(exclude_none=True) if self.quota_store else None
        return BudgetConfig(
            max_tokens=self.max_tokens, max_cost_usd=self.max_cost_usd,
            max_seconds=self.max_seconds, max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls, quota=self.quota, quota_store=store,
        )

    @classmethod
    def _from_config(cls, config: MiddlewareConfig) -> BudgetMiddleware:
        config = BudgetConfig.model_validate(config.model_dump())
        store = CoreQuotaStore.deserialize(config.quota_store) if config.quota_store else None
        return cls(
            **config.model_dump(exclude={"quota", "quota_store"}),
            quota=config.quota, quota_store=store,
        )

    # -------- STATE -----------------------------------------------------------
    def spent(self, ctx: RunContext) -> dict[str, t.Any]:
        """What the current task has used: tokens, cost_usd, seconds,
        model_calls, tool_calls (for the host: billing, dashboards)."""
        state = self.state(ctx)
        internal = ("segment_start", "blocked", "quota_used")
        return {key: value for key, value in state.items() if key not in internal}

    async def quota_used(self, user_id: str) -> QuotaUsage:
        """What ``user_id`` used in the current quota period."""
        if self.quota is None or self.quota_store is None:
            raise ValueError("This budget has no quota")
        return await self.quota_store.usage(user_id, self.quota.period_key(_now()))

    @staticmethod
    def _spent_seconds(state: dict[str, JsonValue]) -> float:
        start = state.get("segment_start")
        running = time.monotonic() - start if isinstance(start, float) else 0.0
        return _number(state, "seconds") + running

    def _exhausted(self, state: dict[str, JsonValue]) -> str | None:
        """Which limit is used up, said for the user; ``None`` if none."""
        blocked = state.get("blocked")
        if isinstance(blocked, str):  # a tool was refused: stop before asking the model again
            return blocked
        checks = (
            (self.max_tokens, _number(state, "tokens"), "token budget ({limit:,} tokens) was used up: {used:,.0f}"),
            (self.max_cost_usd, _number(state, "cost_usd"), "cost budget (${limit:.2f}) was used up: ${used:.4f}"),
            (self.max_seconds, self._spent_seconds(state), "time budget ({limit:.0f}s) was used up: {used:.0f}s"),
            (self.max_model_calls, _number(state, "model_calls"), "model-call budget ({limit}) was used up"),
        )
        for limit, used, text in checks:
            if limit is not None and used >= limit:
                return "This task's " + text.format(limit=limit, used=used) + "."
        if self.quota is not None and "quota_used" in state:
            return self.quota.exceeded(QuotaUsage.model_validate(state["quota_used"]), _now())
        return None

    async def _charge(self, ctx: RunContext, delta: QuotaUsage) -> None:
        """Add to the user's quota and keep the new total for the checks."""
        if self.quota is None or self.quota_store is None:
            return
        key = self.quota.period_key(_now())
        total = await self.quota_store.add(_user(ctx), key, delta)
        self.state(ctx)["quota_used"] = total.model_dump()

    def _before_model(self, ctx: RunContext, call: ModelCall) -> None:
        """No call once a limit is used up. A cost limit without the model's
        prices stops the turn: it never spends without counting."""
        needs_prices = self.max_cost_usd is not None or (self.quota and self.quota.max_cost_usd)
        if needs_prices and _prices(call.model_config) is None:
            raise self.stop(
                "A cost limit needs input_cost_per_mtok and output_cost_per_mtok "
                f"in the ModelConfig of {call.model!r}",
                EXCEEDED,
            )
        state = self.state(ctx)
        if (reason := self._exhausted(state)) is not None:
            raise self.stop(reason, EXCEEDED)
        state["model_calls"] = int(_number(state, "model_calls")) + 1

    async def _after_model(self, ctx: RunContext, call: ModelCall, usage: Usage) -> None:
        """What the call used, added to the task and to the user's quota."""
        state = self.state(ctx)
        tokens = (usage.tokens_input or 0) + (usage.tokens_output or 0)
        cost = 0.0
        if (prices := _prices(call.model_config)) is not None:
            cost = ((usage.tokens_input or 0) * prices[0] + (usage.tokens_output or 0) * prices[1]) / 1_000_000
        state["tokens"] = int(_number(state, "tokens")) + tokens
        state["cost_usd"] = _number(state, "cost_usd") + cost
        await self._charge(ctx, QuotaUsage(tokens=tokens, cost_usd=cost))

    # -------- THE FOUR PLACES ----------------------------------------------------------
    async def turn(self, ctx: RunContext, call_next: NextTurn) -> AsyncGenerator[CoreEvent]:
        # 1. A new task (the user said something) starts from zero; a turn that
        #    goes on after an approval keeps counting.
        state = self.state(ctx)
        new_task = bool(ctx.inbox)
        if new_task:
            state.clear()
        state["segment_start"] = time.monotonic()
        try:
            # 2. The user's quota, before the task starts.
            if self.quota is not None and self.quota_store is not None:
                used = await self.quota_used(_user(ctx))
                state["quota_used"] = used.model_dump()
                if new_task and (reason := self.quota.exceeded(used, _now(), starting_task=True)):
                    raise self.stop(reason, EXCEEDED)
            if (reason := self._exhausted(state)) is not None:
                raise self.stop(reason, EXCEEDED)
            if new_task:
                await self._charge(ctx, QuotaUsage(tasks=1))
            # 3. The turn; its time counts only while it runs.
            async for event in call_next():
                yield event
        finally:
            state["seconds"] = self._spent_seconds(state)
            state.pop("segment_start", None)

    async def model(self, ctx: RunContext, call: ModelCall, call_next: NextModel) -> LLMResponse:
        self._before_model(ctx, call)
        result = await call_next(call)
        await self._after_model(ctx, call, result.usage)
        return result

    async def stream(
        self, ctx: RunContext, call: ModelCall, call_next: NextStream
    ) -> AsyncGenerator[LLMChunk]:
        self._before_model(ctx, call)
        async for chunk in call_next(call):
            if chunk.is_complete:
                await self._after_model(ctx, call, chunk.usage or Usage())
            yield chunk

    async def tool(self, ctx: RunContext, record: ToolCallRecord, call_next: NextTool) -> ToolResult | None:
        # 1. A limit used up: the call answers "not run", so the transcript stays whole.
        state = self.state(ctx)
        reason = self._exhausted(state)
        calls = _number(state, "tool_calls")
        first_pass = record.status == ToolCallStatus.PENDING_APPROVAL
        if reason is None and first_pass and self.max_tool_calls is not None and calls >= self.max_tool_calls:
            reason = f"This task's tool-call budget ({self.max_tool_calls}) was used up."
        if reason is not None:
            state["blocked"] = reason
            return ToolResult.execution_error(record.id, f"Not run: {reason}")
        # 2. Counted once: a call that waits for approval passes here twice.
        if first_pass:
            state["tool_calls"] = int(calls) + 1
        return await call_next(record)


def _user(ctx: RunContext) -> str:
    if not ctx.user_id:
        raise ValueError("A user quota needs RunContext.user_id")
    return ctx.user_id


def _number(state: dict[str, JsonValue], key: str) -> float:
    """A count kept in the state; 0 when there is none yet."""
    value = state.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _prices(config: ModelConfig | None) -> tuple[float, float] | None:
    if config is None or config.input_cost_per_mtok is None or config.output_cost_per_mtok is None:
        return None
    return config.input_cost_per_mtok, config.output_cost_per_mtok


__all__ = ["BudgetConfig", "BudgetMiddleware"]
