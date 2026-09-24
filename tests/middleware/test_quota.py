"""A user's quota is shared by all their tasks, in any process, per period."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.middleware import BudgetMiddleware
from max_ai.capabilities.middleware import budget as budget_module
from max_ai.capabilities.quota_store import LocalQuotaStore
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.model.quota import QuotaLimits, QuotaUsage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext

DAY = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)


class FakeLLM:
    """Every call answers and costs 1,000 + 100 tokens."""

    model = "fake"

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.generation_options = {"max_tokens": 500}
        self.calls = 0

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.calls += 1
        return ChatCompletionResult(
            message=AssistantMessage(source="llm", content=f"respuesta {self.calls}"),
            usage=Usage(tokens_input=1000, tokens_output=100), model="fake", finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def fixed_day(monkeypatch):
    monkeypatch.setattr(budget_module, "_now", lambda: DAY)


def make_agent(tmp_path, llm, quota: QuotaLimits) -> Agent:
    budget = BudgetMiddleware(quota=quota, quota_store=LocalQuotaStore(tmp_path / "quota"))
    return Agent(name="a", description="d", instructions="i", client=llm,
                 workspace=LocalWorkspace(root=tmp_path / "work"), middlewares=[budget])


async def ask(agent: Agent, user: str, text: str = "hola"):
    async with agent:
        return await agent.run(text, run_context=RunContext(user_id=user))


async def test_the_quota_is_shared_by_every_task_and_agent_instance(tmp_path):
    quota = QuotaLimits(max_tokens=2500)
    llm = FakeLLM()
    # A new Agent per request, like a serverless host: the usage lives in the store.
    results = [await ask(make_agent(tmp_path, llm, quota), "ana") for _ in range(4)]
    assert [r.finish_reason for r in results] == ["stop", "stop", "stop", "budget_exceeded"]
    assert llm.calls == 3  # the 4th task never reached the model
    assert "daily quota (2,500 tokens) is used up; it resets at 2026-09-24 00:00 UTC" in results[3].stop_message

    other = await ask(make_agent(tmp_path, llm, quota), "beto")  # another user: own quota
    assert other.finish_reason == "stop"


async def test_task_limit_and_period_reset(tmp_path, monkeypatch):
    quota = QuotaLimits(max_tasks=2)
    llm = FakeLLM()
    first, second, third = [await ask(make_agent(tmp_path, llm, quota), "ana") for _ in range(3)]
    assert (first.finish_reason, second.finish_reason) == ("stop", "stop")
    assert third.finish_reason == "budget_exceeded" and "2 tasks" in third.stop_message

    monkeypatch.setattr(budget_module, "_now", lambda: DAY + timedelta(days=1))
    assert (await ask(make_agent(tmp_path, llm, quota), "ana")).finish_reason == "stop"

    saved = json.loads((tmp_path / "quota" / "ana.json").read_text())
    assert saved["day:2026-09-23"]["tasks"] == 2 and saved["day:2026-09-24"]["tasks"] == 1


async def test_monthly_cost_quota_uses_the_model_prices(tmp_path):
    prices = ModelConfig(input_cost_per_mtok=2.0, output_cost_per_mtok=10.0)
    agent = make_agent(tmp_path, FakeLLM(prices), QuotaLimits(period="month", max_cost_usd=5.0))
    await ask(agent, "ana")
    used = await agent.middlewares[0].quota_used("ana")
    assert used.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert used.tokens == 1100 and used.tasks == 1

    with pytest.raises(ValueError, match="input_cost_per_mtok"):
        await ask(make_agent(tmp_path, FakeLLM(), QuotaLimits(max_cost_usd=5.0)), "ana")


async def test_concurrent_charges_are_not_lost(tmp_path):
    store = LocalQuotaStore(tmp_path)
    await asyncio.gather(*(store.add("ana", "day:x", QuotaUsage(tokens=10)) for _ in range(50)))
    assert (await store.usage("ana", "day:x")).tokens == 500


def test_budget_with_quota_travels_with_the_agent(tmp_path):
    agent = make_agent(tmp_path, FakeLLM(), QuotaLimits(period="month", max_tokens=10_000))
    spec = agent.middlewares[0].serialize()
    back = BudgetMiddleware.deserialize(spec.model_dump_json())
    assert back.quota == QuotaLimits(period="month", max_tokens=10_000)
    assert back.quota_store.base_path == tmp_path / "quota"

    with pytest.raises(ValueError, match="go together"):
        BudgetMiddleware(quota=QuotaLimits(max_tokens=1))
