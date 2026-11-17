"""A coding agent: clones a repository, works on its own branch and runs the
tests, from the CLI. Try: "clone <repo url>, fix <bug> on a new branch and run
the tests"."""

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
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.capabilities.session_store.local import LocalSessionStore
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.tools.bash import BashTool
from max_ai.cli import run_cli
from max_ai.types.tools import ToolApprovalMode

HERE = Path(__file__).parent

# Cheap paid OpenRouter models (free ones hang), with fallbacks.
_client: dict[str, t.Any] = {
    "model": "google/gemini-4-hydrogen",
    "fallback_models": ["openai/gpt-5.6-luna", "anthropic/claude-haiku-5.5",],
    "app_name": "coding_agent",
}


async def main() -> None:
    load_dotenv()
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None

    agent = Agent(
        name="Coder",
        description="Writes and fixes code in git repositories, and runs their tests.",
        instructions=(
            "You help users with code. Read the code before you change it, keep "
            "changes small, and run the tests before you say a task is done. "
            "If the project is not a git repository, run git init and commit when "
            "a step works. For risky or long changes on an existing project, use "
            "enter_worktree."
        ),
        client=OpenRouterChatCompletionClient(**_client),
        toolset=[BashTool()],  # with bash the harness adds enter_worktree and exit_worktree
        mcp=[
            # web search, for documentation
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                headers_env={"x-api-key": "EXA_API_KEY"} if os.getenv("EXA_API_KEY") else {},
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
        skills=LocalSkills(str(HERE / "skills"), ["qr-code"]),  # needs pip install to run
        memory=Memory(),
        # A container per user: it never sees this machine's files or .env.
        sandbox=DockerSandbox(
            dockerfile=str(HERE / "coding.Dockerfile"),  # python and git
            networks=["pypi.org", "files.pythonhosted.org", "github.com"],  # pip and clones
        ),
        middlewares=[TracingMiddleware()] if tracing else [],
    )

    # The whole agent as YAML, with no secrets: Agent.from_yaml(path) builds it again.
    saved = HERE / "agents" / "coding_agent.yaml"
    saved.parent.mkdir(exist_ok=True)
    saved.write_text(agent.to_yaml())

    try:
        # Every turn is saved on disk; /resume continues one.
        store = LocalSessionStore(HERE / "coding_sessions")
        await run_cli(agent, user_id="demo-coding", store=store)
    finally:
        await agent.close()
        if tracing:
            tracing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
