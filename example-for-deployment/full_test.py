"""The new agent end to end: web search (MCP), a skill from GitHub, a sandbox,
memory and tracing, run from the CLI."""

import asyncio
import os
import typing as t
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.mcp import HTTPServerConfig
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.session_store.local import LocalSessionStore
from max_ai.capabilities.skills import GitHubSkills
from max_ai.capabilities.tools.bash import BashTool
from max_ai.cli import run_cli
from max_ai.types.tools import ToolApprovalMode

HERE = Path(__file__).parent
ROOT = HERE / "full_test_box"  # the sandbox: files, skills, memory and setting.json

# Cheap paid OpenRouter models (free ones hang), with fallbacks.
_client: dict[str, t.Any] = {
    "model": "anthropic/claude-haiku-5.5",
    "fallback_models": ["openai/gpt-5.6-luna", "qwen/qwen4-max"],
    "app_name": "full_test",
}


async def main() -> None:
    load_dotenv()
    ROOT.mkdir(exist_ok=True)
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    agent = Agent(
        name="TravelPlanner",
        description="Researches trips on the web and delivers the plan as an Excel workbook.",
        instructions="You help users plan trips.",
        client=OpenRouterChatCompletionClient(**_client),
        toolset=[BashTool()],  # skills run their scripts with bash
        mcp=[
            # web search
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                headers_env={"x-api-key": "EXA_API_KEY"} if os.getenv("EXA_API_KEY") else {},
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
        skills=GitHubSkills(  # someone else's skill, straight from GitHub
            "https://github.com/trailofbits/skills-curated/tree/main/plugins/openai-spreadsheet/skills",
            ["openai-spreadsheet"],
        ),
        memory=Memory(),
        sandbox=LocalSandbox(root=str(ROOT)),
        middlewares=[TracingMiddleware()] if tracing else [],
    )

    # The whole agent as YAML, with no secrets: Agent.from_yaml(path) builds it again.
    saved = HERE / "agents" / "full_test.yaml"
    saved.parent.mkdir(exist_ok=True)
    saved.write_text(agent.to_yaml())

    try:
        # Every turn is saved on disk; /resume continues one.
        store = LocalSessionStore(HERE / "full_test_sessions")
        await run_cli(agent, user_id="demo-full-test", store=store)
    finally:
        await agent.close()
        if tracing:
            tracing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
