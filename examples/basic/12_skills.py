"""A skill from GitHub: instructions and scripts the model reads when a task
needs them. Skills need a sandbox to live in and bash to run their scripts.
Set GITHUB_TOKEN if GitHub answers 403: without one its rate limit is low."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills import GitHubSkills
from max_ai.capabilities.tools.bash import BashTool
from max_ai.types.run_context import RunContext

ROOT = Path("./local/box")


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    skills = GitHubSkills(
        "https://github.com/trailofbits/skills-curated/tree/main/plugins/openai-spreadsheet/skills",
        ["openai-spreadsheet"],  # only the ones named are installed
    )
    agent = Agent(
        name="Analyst",
        description="An assistant that builds spreadsheets.",
        instructions="Help the user with data and spreadsheets.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        toolset=[BashTool()],
        skills=skills,
        sandbox=LocalSandbox(root=str(ROOT)),
    )
    ctx = RunContext(user_id="ana", session_id="budget")
    try:
        state = await agent.run("Create budget.xlsx with a monthly budget for a student.", ctx)
        # bash waits for approval by default: see 04_tool_approval.py to go on from here.
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
