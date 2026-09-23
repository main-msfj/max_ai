"""Budget for one task and quota per user: tokens, cost, time and calls.

Per task (``max_*``): a task is what one new input starts; resuming it
after an approval or a question keeps counting. Time counts only while
the agent runs, never while it waits for the user.

Per user (``quota`` + ``quota_store``): limits per day or month shared by
all the user's tasks, counted in a ``CoreQuotaStore`` so every process and
every serverless invocation sees the same usage. It is charged after each
model call, so concurrent tasks of one user may go slightly over: the
next check stops them.

When a limit is reached:
- the next model call does not happen: the run ends with
  ``finish_reason="budget_exceeded"`` and a message for the user;
- tools the model already asked for do not run: they answer
  "not run, budget used up", so the transcript stays consistent.
"""

from __future__ import annotations

import time
import typing as t
from datetime import datetime, timezone

from pydantic import Field, model_validator

from ...base.middleware import (
    CoreMiddleware,
    MiddlewareConfig,
    MiddlewareContext,
    ModelRequest,
    StopRun,
    ToolRequest,
)
from ...base.quota_store import CoreQuotaStore
from ...core.model.quota import QuotaLimits, QuotaUsage
from ...types.tool_call import ToolResult

if t.TYPE_CHECKING:
    from ...core.messages import CoreMessage
    from ...types.agent_response import AgentResponse
    from ...types.completions import ChatCompletionResult
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
    def _from_config(cls, config: BudgetConfig) -> BudgetMiddleware:
        store = CoreQuotaStore.deserialize(config.quota_store) if config.quota_store else None
        return cls(
            **config.model_dump(exclude={"quota", "quota_store"}),
            quota=config.quota, quota_store=store,
        )

    # -------- STATE -----------------------------------------------------------
    def spent(self, ctx: RunContext) -> dict[str, t.Any]:
        """What the current task has used: tokens, cost_usd, seconds,
        model_calls, tool_calls (for the host: billing, dashboards)."""
        state = ctx.runtime_state.shared_state.get("middleware", {}).get(self.state_key, {})
        internal = ("segment_start", "blocked", "quota_used")
        return {key: value for key, value in state.items() if key not in internal}

    async def quota_used(self, user_id: str) -> QuotaUsage:
        """What ``user_id`` used in the current quota period."""
        if self.quota is None or self.quota_store is None:
            raise ValueError("This budget has no quota")
        return await self.quota_store.usage(user_id, self.quota.period_key(_now()))

    def _spent_seconds(self, state: dict[str, t.Any]) -> float:
        running = time.monotonic() - state["segment_start"] if state.get("segment_start") else 0
        return state.get("seconds", 0.0) + running

    def _exhausted(self, state: dict[str, t.Any]) -> str | None:
        """Which limit is used up, said for the user; ``None`` if none."""
        if state.get("blocked"):  # a tool was refused: stop before asking the model again
            return state["blocked"]
        checks = (
            (self.max_tokens, state.get("tokens", 0), "token budget ({limit:,} tokens) was used up: {used:,}"),
            (self.max_cost_usd, state.get("cost_usd", 0.0), "cost budget (${limit:.2f}) was used up: ${used:.4f}"),
            (self.max_seconds, self._spent_seconds(state), "time budget ({limit:.0f}s) was used up: {used:.0f}s"),
            (self.max_model_calls, state.get("model_calls", 0), "model-call budget ({limit}) was used up"),
        )
        for limit, used, text in checks:
            if limit is not None and used >= limit:
                return "This task's " + text.format(limit=limit, used=used) + "."
        if self.quota is not None and "quota_used" in state:
            return self.quota.exceeded(QuotaUsage.model_validate(state["quota_used"]), _now())
        return None

    async def _charge(self, mw: MiddlewareContext, delta: QuotaUsage) -> None:
        """Add to the user's quota and keep the new total for the checks."""
        if self.quota is None or self.quota_store is None:
            return
        key = self.quota.period_key(_now())
        total = await self.quota_store.add(_user(mw), key, delta)
        mw.state(self)["quota_used"] = total.model_dump()

    # -------- HOOKS -----------------------------------------------------------
    async def on_run_start(self, mw: MiddlewareContext, task: list[CoreMessage] | None) -> None:
        state = mw.state(self)
        if task is not None:  # a new task starts from zero; a resume keeps counting
            state.clear()
        state["segment_start"] = time.monotonic()
        if self.quota is not None and self.quota_store is not None:
            used = await self.quota_used(_user(mw))
            state["quota_used"] = used.model_dump()
            if task is not None and (reason := self.quota.exceeded(used, _now(), starting_task=True)):
                raise StopRun(reason, finish_reason=EXCEEDED)
        if (reason := self._exhausted(state)) is not None:
            raise StopRun(reason, finish_reason=EXCEEDED)
        if task is not None:
            await self._charge(mw, QuotaUsage(tasks=1))

    async def on_run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        state = mw.state(self)
        state["seconds"] = self._spent_seconds(state)
        state.pop("segment_start", None)

    async def on_model_request(self, mw: MiddlewareContext, request: ModelRequest) -> ModelRequest:
        needs_prices = self.max_cost_usd is not None or (self.quota and self.quota.max_cost_usd)
        if needs_prices and _prices(request) is None:
            raise ValueError(
                "A cost limit needs input_cost_per_mtok and output_cost_per_mtok "
                f"in the ModelConfig of {request.model!r}"
            )
        state = mw.state(self)
        if (reason := self._exhausted(state)) is not None:
            raise StopRun(reason, finish_reason=EXCEEDED)
        state["model_calls"] = state.get("model_calls", 0) + 1
        return request

    async def on_model_response(
        self, mw: MiddlewareContext, request: ModelRequest, result: ChatCompletionResult,
    ) -> ChatCompletionResult:
        state = mw.state(self)
        tokens = (result.usage.tokens_input or 0) + (result.usage.tokens_output or 0)
        cost = 0.0
        if (prices := _prices(request)) is not None:
            cost = ((result.usage.tokens_input or 0) * prices[0]
                    + (result.usage.tokens_output or 0) * prices[1]) / 1_000_000
        state["tokens"] = state.get("tokens", 0) + tokens
        state["cost_usd"] = state.get("cost_usd", 0.0) + cost
        await self._charge(mw, QuotaUsage(tokens=tokens, cost_usd=cost))
        return result

    async def on_tool_request(self, mw: MiddlewareContext, request: ToolRequest) -> ToolResult | None:
        state = mw.state(self)
        reason = self._exhausted(state)
        calls = state.get("tool_calls", 0)
        if reason is None and self.max_tool_calls is not None and calls >= self.max_tool_calls:
            reason = f"This task's tool-call budget ({self.max_tool_calls}) was used up."
        if reason is not None:
            state["blocked"] = reason
            return ToolResult.execution_error(request.record.id, f"Not run: {reason}")
        state["tool_calls"] = calls + 1
        return None


def _user(mw: MiddlewareContext) -> str:
    if not mw.ctx.user_id:
        raise ValueError("A user quota needs RunContext.user_id")
    return mw.ctx.user_id


def _prices(request: ModelRequest) -> tuple[float, float] | None:
    config = request.model_config
    if config is None or config.input_cost_per_mtok is None or config.output_cost_per_mtok is None:
        return None
    return config.input_cost_per_mtok, config.output_cost_per_mtok


__all__ = ["BudgetConfig", "BudgetMiddleware"]
