"""A bookkeeper: closes the month of a small business and hands in the report
as a web page and as an Excel workbook. Try: "close October for me"."""

import asyncio
import os
import typing as t
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.mcp import HTTPSecret, HTTPServerConfig
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.session_store.local import LocalSessionStore
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.stacks import CustomLayer
from max_ai.capabilities.tools.bash import BashTool
from max_ai.cli import run_cli
from max_ai.types.tools import ToolApprovalMode

HERE = Path(__file__).parent
BOX = HERE / "box"  # the sandbox: the work of each session, the skills and the memory

# A free model first; the paid ones answer if it does not.
_client: dict[str, t.Any] = {
    "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "fallback_models": ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5"],
    "app_name": "bookkeeper",
    # Claude uses the provider's cache only when asked; other models ignore this.
    "options": {"cache_control": {"type": "ephemeral"}},
}

# How it behaves, in a few lines: the harness allows no more.
INSTRUCTIONS = """You are the bookkeeper of a small business.
First write your plan with write_todos. Do the sums with a script. Ask about
an entry you cannot classify and look up exchange rates on the web. Save what
the user prefers with save_memory. Hand in a web page and a workbook with
your skills."""

# What this business is like goes in a prompt layer of its own.
BUSINESS = f"""The business is Café La Esquina, in Guatemala. The books of each
month are CSV files in {HERE / "data"}: copy them into your working directory
before you read them, and never change the originals."""


async def main() -> None:
    load_dotenv()
    BOX.mkdir(exist_ok=True)
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    agent = Agent(
        name="Bookkeeper",
        description="Closes the month of a small business and writes its report.",
        instructions=INSTRUCTIONS,
        client=OpenRouterChatCompletionClient(**_client),
        toolset=[BashTool()],  # the sums and the skills' scripts; the harness adds the rest
        mcp=[
            # Web search, for the exchange rate. The key stays in the environment:
            # the config keeps only its name, so the YAML below holds no secret.
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                secret=HTTPSecret(
                    type="api_key", header="x-api-key",
                    keys={"EXA_API_KEY": os.getenv("EXA_API_KEY")},
                ),
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
        skills=LocalSkills(str(HERE / "skills"), ["html-report", "excel-report"]),
        prompt_layers=[CustomLayer(content=BUSINESS)],
        memory=Memory(),  # what the user likes is kept for the next month
        sandbox=LocalSandbox(root=str(BOX)),  # this computer: easy to look at, not isolation
        middlewares=[TracingMiddleware()] if tracing else [],
    )

    # The whole agent as YAML, with no secrets: Agent.from_yaml(path) builds it again.
    (HERE / "agent.yaml").write_text(agent.to_yaml())

    try:
        # Every turn is saved on disk; /resume continues one.
        store = LocalSessionStore(HERE / "sessions")
        await run_cli(agent, user_id="demo", store=store)
    finally:
        await agent.close()
        if tracing:
            tracing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
