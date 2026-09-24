"""A travel planner: searches the web and delivers the trip as an Excel file.

    .venv/bin/python example-for-deployment/travel_agent.py              # run it
    .venv/bin/python example-for-deployment/travel_agent.py --save       # write travel_agent.json
    .venv/bin/python example-for-deployment/travel_agent.py --from-json  # run the saved agent

The JSON is the whole agent (model, MCP, skills, workspace, executor,
middleware) with no secrets: only the names of the env vars that hold them.
Store it in a database and any server can rebuild the same agent.

Needs in .env: OPENROUTER_API_KEY, MONGODB_URI, MINIO_ACCESS_KEY=maxai and
MINIO_SECRET_KEY=maxai-local-dev (docker-infra/capabilities MinIO), and
`modal setup` done once. EXA_API_KEY is optional (Exa works without one, with
lower limits). With LANGFUSE_PUBLIC_KEY/SECRET_KEY set, every turn is a trace.
"""

import asyncio
import os
import sys
import typing as t
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.executor.modal import ModalExecutor
from max_ai.capabilities.mcp import HTTPServerConfig
from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
from max_ai.capabilities.session_store import MongoDBSessionStore
from max_ai.capabilities.skills.github import GithubSkillRegistry
from max_ai.capabilities.workspace import MinIOWorkspace
from max_ai.cli import run_cli
from max_ai.types.tools import ToolApprovalMode

# PreConfig Vales
_client: dict[str, t.Any] = {
    "model": "nvidia/nemotron-3-super-120b-a12b:free",
    "fallback_models": ["qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"],
    "reasoning": {"effort": "low"},
    "app_name": "travel_agent",
}


CONFIG = Path(__file__).with_name("travel_agent.json")


def build_agent(tracing: bool) -> Agent:
    return Agent(
        name="TravelPlanner",
        description="Plans trips working for Sara trips",
        instructions="Help the user plan trips. user your tools and skill at will",
        client=OpenRouterChatCompletionClient(**_client),
        mcp=[
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                headers_env={"x-api-key": "EXA_API_KEY"}
                if os.getenv("EXA_API_KEY")
                else {},
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
        skills=GithubSkillRegistry(
            "trailofbits/skills-curated",
            ["openai-spreadsheet"],
            path="plugins/openai-spreadsheet/skills",
        ),
        workspace=MinIOWorkspace(
            os.getenv("MINIO_ENDPOINT", "http://localhost:9000"), bucket="workspaces"
        ),
        executor=ModalExecutor(packages=["openpyxl", "pandas"]),
        middlewares=[TracingMiddleware()] if tracing else [],
    )


async def main() -> None:
    load_dotenv()
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    if "--from-json" in sys.argv:
        agent = Agent.deserialize(CONFIG.read_text())
    else:
        agent = build_agent(tracing is not None)
    if "--save" in sys.argv:
        CONFIG.write_text(agent.serialize().model_dump_json(indent=2))
        print(f"Saved {CONFIG}")
        return

    try:
        async with agent:
            # Every turn is saved in MongoDB (see it in Mongo Express); /resume continues one.
            await run_cli(agent, user_id="demo", store=MongoDBSessionStore())
    finally:
        if tracing:
            tracing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
