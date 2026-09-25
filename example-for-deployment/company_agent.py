"""Acme AI's company assistant: HR policies, vacation and parking.

Start the HR MCP server first (example-for-deployment/mcps/acme_company).
Needs in .env: OPENAI_API_KEY, MONGODB_URI, MINIO_ACCESS_KEY/MINIO_SECRET_KEY
and ACME_MCP_TOKEN (the employee's JWT). With LANGFUSE_* set, every turn is a trace.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.executor.modal import ModalExecutor
from max_ai.capabilities.knowledge import MongoDBKnowledgeRegistry
from max_ai.capabilities.mcp import HTTPServerConfig
from max_ai.capabilities.memory import MongoDBMemoryRegistry
from max_ai.capabilities.middleware import (
    BudgetMiddleware,
    TracingMiddleware,
    configure_langfuse,
)
from max_ai.capabilities.session_store import MongoDBSessionStore
from max_ai.capabilities.skills.github import GithubSkillRegistry
from max_ai.capabilities.workspace import MinIOWorkspace
from max_ai.cli import run_cli
from max_ai.core.blocks import KnowledgeBlock

DATA = Path(__file__).parent / "data"
EMPLOYEE_ID = "0123456789"


# -------- data: the employee's memory and the HR policy, kept apart from the agent
async def load_memory(memory: MongoDBMemoryRegistry) -> None:
    profile = memory.bind(EMPLOYEE_ID, "profile")
    for record in json.loads((DATA / "memory.json").read_text()).values():
        await profile.create_or_update(record["category"], record["memory"])


async def load_knowledge(knowledge: MongoDBKnowledgeRegistry) -> None:
    for block in json.loads((DATA / "knowledge.json").read_text()):
        meta = block["metadata"]
        await knowledge.upsert_block(f"{meta['policy']}-{meta['section']}", KnowledgeBlock(**block))


async def main() -> None:
    load_dotenv()
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    memory = MongoDBMemoryRegistry()
    hr_policies = MongoDBKnowledgeRegistry(
        name="hr_policies",
        description="Acme AI HR policies: vacation, parking, sick leave, remote work, "
                    "parental leave, expenses and public holidays.",
    )
    await load_memory(memory)
    await load_knowledge(hr_policies)

    agent = Agent(
        name="AcmeAssistant",
        description="Acme AI's company assistant.",
        instructions="You help Acme AI employees with company policies, vacation and parking.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", reasoning_effort="none"),
        memory=memory,  # who the employee is, across conversations
        knowledge=[hr_policies],  # the HR policy, searched by meaning
        mcp=[  # the HR system; the token says which employee is calling
            HTTPServerConfig(
                server_id="acme_hr",
                url=os.getenv("ACME_MCP_URL", "http://127.0.0.1:8765/mcp"),
                token_env="ACME_MCP_TOKEN",
            )
        ],
        workspace=MinIOWorkspace(
            os.getenv("MINIO_ENDPOINT", "http://localhost:9000"), bucket="workspaces"
        ),
        skills=GithubSkillRegistry(  # Word documents, same repo as the travel agent's skill
            "trailofbits/skills-curated",
            ["openai-doc"],
            path="plugins/openai-doc/skills",
        ),
        executor=ModalExecutor(),  # isolated sandbox; can only reach pip/npm registries
        middlewares=[
            BudgetMiddleware(max_tokens=200_000, max_tool_calls=30),
            *([TracingMiddleware()] if tracing else []),
        ],
    )

    try:
        async with agent:
            await run_cli(agent, user_id=EMPLOYEE_ID, store=MongoDBSessionStore())
    finally:
        if tracing:
            tracing.shutdown()

    # The whole agent as JSON, with no secrets: any server can rebuild it.
    saved = Path(__file__).parent / "agents" / "company_agent.json"
    saved.write_text(agent.serialize().model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
