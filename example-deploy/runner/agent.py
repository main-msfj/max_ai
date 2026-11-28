"""A test runner: starts a slow test suite in the background and keeps helping
meanwhile. Try: "run the tests, and meanwhile search what is new in pytest"."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents.agent import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.mcp import HTTPSecret, HTTPServerConfig
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.session_store.local import LocalSessionStore
from max_ai.capabilities.skills import LocalSkills
from max_ai.capabilities.tools.bash import BashTool
from max_ai.cli import run_cli
from max_ai.types.tools import ToolApprovalMode

HERE = Path(__file__).parent
BOX = HERE / "box"  # the sandbox: the work of each session, the skills and the tasks

INSTRUCTIONS = """You help a developer with their project.
A command that takes more than a few seconds runs in the background: say it
started and go on with what else was asked. When you are told it ended, report
its result. Look things up on the web when asked."""


async def main() -> None:
    load_dotenv()
    BOX.mkdir(exist_ok=True)

    agent = Agent(
        name="Runner",
        description="Runs slow commands in the background and keeps helping meanwhile.",
        instructions=INSTRUCTIONS,
        client=OpenRouterChatCompletionClient(
            model="deepseek/deepseek-v4-flash", app_name="runner"
        ),
        toolset=[BashTool()],  # with it come task_output and stop_task
        mcp=[
            # Web search. The key stays in the environment: the config keeps only its name.
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
        skills=LocalSkills(str(HERE / "skills"), ["slow-tests"]),
        sandbox=LocalSandbox(root=str(BOX)),  # this computer: easy to look at, not isolation
    )

    try:
        # Every turn is saved on disk; /resume continues one.
        await run_cli(agent, user_id="demo", store=LocalSessionStore(HERE / "sessions"))
    finally:
        await agent.close()  # stops the tasks still running


if __name__ == "__main__":
    asyncio.run(main())
