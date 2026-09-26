"""A travel planner"""

import asyncio
import os
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

# Free OpenRouter models, with fallbacks when one is busy.
_client: dict[str, t.Any] = {
    "model": "nvidia/nemotron-3-super-120b-a12b:free",
    "fallback_models": ["qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"],
    "reasoning": {"effort": "low"},
    "app_name": "travel_agent",
}


async def main() -> None:
    load_dotenv()
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    agent = Agent(
        name="TravelPlanner",
        description="Researches trips on the web and delivers the plan as an Excel workbook.",
        instructions="You help users plan trips.",
        client=OpenRouterChatCompletionClient(**_client),
        mcp=[ 
            # web search
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                headers_env={"x-api-key": "EXA_API_KEY"} if os.getenv("EXA_API_KEY") else {},
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
        skills=GithubSkillRegistry(  # someone else's skill, straight from GitHub
            "trailofbits/skills-curated",
            ["openai-spreadsheet"],
            path="plugins/openai-spreadsheet/skills",
        ),
        workspace=MinIOWorkspace(  # the user's files, in object storage
            os.getenv("MINIO_ENDPOINT", "http://localhost:9000"), bucket="workspaces"
        ),
        executor=ModalExecutor(),  # isolated sandbox; can only reach pip/npm registries
        middlewares=[TracingMiddleware()] if tracing else [],
    )

    try:
        async with agent:
            # Every turn is saved in MongoDB; /resume continues one.
            await run_cli(agent, user_id="demo-1-travel", store=MongoDBSessionStore())
    finally:
        if tracing:
            tracing.shutdown()

    # The whole agent as JSON, with no secrets: any server can rebuild it.
    saved = Path(__file__).parent / "agents" / "travel_agent.json"
    saved.write_text(agent.serialize().model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
