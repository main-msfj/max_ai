"""The production stack, in the CLI: MongoDB memory, knowledge and quotas,
plus traces in Langfuse. Start the services first (see docker-infra/README.md):

    cd docker-infra && docker compose up -d --wait mongodb mongo-express
    docker compose --profile observability up -d --wait

Then, from the repository root, with OPENAI_API_KEY in .env:
    .venv/bin/python -m examples.03_agent_with_mongodb [--session <id>]

MONGODB_URI defaults to the docker-infra MongoDB; the Langfuse keys default
to the ones docker-infra creates. Browse the data at http://localhost:8081
(Mongo Express) and the traces at http://localhost:3000 (Langfuse).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from examples.shared import LOCAL_DIR, USER_ID, get_weather, session_arg
from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry
from max_ai.capabilities.middleware import (
    BudgetMiddleware,
    TracingMiddleware,
    configure_langfuse,
)
from max_ai.capabilities.quota_store import MongoDBQuotaStore
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.cli import run_cli
from max_ai.core import KnowledgeBlock
from max_ai.core.embeddings import FastEmbedEmbedding
from max_ai.core.model.quota import QuotaLimits
from max_ai.types.tools import ToolApprovalMode

MODEL = "gpt-5.6-luna"
# docker-infra defaults; override them in .env.
DEFAULTS = {
    "MONGODB_URI": "mongodb://maxai:maxai-local-dev@localhost:27017/?authSource=admin",
    "LANGFUSE_HOST": "http://localhost:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-maxai-local",
    "LANGFUSE_SECRET_KEY": "sk-lf-maxai-local",
}


async def load_knowledge(knowledge: MongoDBKnowledgeRegistry) -> None:
    """Load the example documents into MongoDB (idempotent: same ids)."""
    blocks = json.loads((LOCAL_DIR / "knowledge" / "framework.json").read_text())
    for index, block in enumerate(blocks):
        await knowledge.upsert_block(
            f"framework-{index}", KnowledgeBlock.model_validate(block)
        )


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Configura OPENAI_API_KEY en el entorno o en .env.")

    client = OpenAIChatCompletionClient(
        model=MODEL, reasoning_effort="none", max_tokens=32_000
    )
    knowledge = MongoDBKnowledgeRegistry(
        name="framework",
        description="How the max_ai framework works: loop, tools, gates, memory, knowledge, skills.",
        embedding=FastEmbedEmbedding(),  # search by meaning, not exact words
    )
    await load_knowledge(knowledge)
    tracing = configure_langfuse()

    agent = Agent(
        name="MongoDemo",
        description="Agente con memoria, knowledge y cuotas en MongoDB, y trazas en Langfuse.",
        instructions="Ayuda al usuario. Guarda en memoria lo importante que te cuente.",
        client=client,
        toolset=[
            FunctionAsTool(
                get_weather,
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
                read_only=True,
            )
        ],
        # One registry for every user: each run binds it to its user and session.
        memory=MongoDBMemoryRegistry(embedding=FastEmbedEmbedding()),
        knowledge=[knowledge],
        compaction=SummaryCompaction(threshold=0.8, keep_ratio=0.4),
        middlewares=[
            # Per task and per user per day; the usage lives in MongoDB.
            BudgetMiddleware(
                max_model_calls=30,
                quota=QuotaLimits(period="day", max_tokens=2_000_000),
                quota_store=MongoDBQuotaStore(),
            ),
            TracingMiddleware(),
        ],
    )
    try:
        async with agent:
            await run_cli(
                agent,
                store=LocalSessionStore(LOCAL_DIR / "sessions"),
                user_id=USER_ID,
                session_id=session_arg(),
            )
    finally:
        await knowledge.disconnect()
        await client.client.close()
        tracing.shutdown()  # sends the spans still buffered


if __name__ == "__main__":
    asyncio.run(main())
