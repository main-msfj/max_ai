"""MongoDBQuotaStore against a real MongoDB (skipped without MONGODB_URI)."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.middleware import BudgetMiddleware
from max_ai.capabilities.quota_store.mongodb import MongoDBQuotaStore
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.core.model.quota import QuotaLimits, QuotaUsage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext

pytestmark = pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="needs MONGODB_URI")


@pytest.fixture
def collection() -> str:
    return f"quota_test_{uuid.uuid4().hex[:8]}"


async def test_charges_from_several_processes_are_never_lost(collection):
    # Two stores = two processes (or Lambdas) sharing the same database.
    first, second = MongoDBQuotaStore(collection=collection), MongoDBQuotaStore(collection=collection)
    async with first, second:
        await asyncio.gather(*(
            store.add("ana", "day:2026-09-23", QuotaUsage(tokens=10, tasks=1))
            for _ in range(100) for store in (first, second)
        ))
        used = await first.usage("ana", "day:2026-09-23")
        assert (used.tokens, used.tasks) == (2000, 200)
        assert await second.usage("beto", "day:2026-09-23") == QuotaUsage()
        await first._collection.drop()


class FakeLLM:
    model = "fake"
    config = ModelConfig()
    generation_options = {"max_tokens": 500}

    async def run(self, **kwargs):
        return ChatCompletionResult(
            message=AssistantMessage(source="llm", content="ok"),
            usage=Usage(tokens_input=1000, tokens_output=100), model="fake", finish_reason="stop",
        )


async def test_a_quota_in_mongodb_stops_new_agent_instances(tmp_path, collection):
    store = MongoDBQuotaStore(collection=collection)
    quota = QuotaLimits(max_tokens=2000)
    finish = []
    for _ in range(3):  # a new Agent per request, as in serverless
        agent = Agent(name="a", description="d", instructions="i", client=FakeLLM(),
                      workspace=LocalWorkspace(root=tmp_path),
                      middlewares=[BudgetMiddleware(quota=quota, quota_store=store)])
        async with agent:
            finish.append((await agent.run("hola", run_context=RunContext(user_id="ana"))).finish_reason)
    assert finish == ["stop", "stop", "budget_exceeded"]
    await store._collection.drop()
    await store.disconnect()
