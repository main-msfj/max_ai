"""Use a skill from a GitHub repository (cloned once, then cached)."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.skills.github import GithubSkillRegistry


async def main() -> None:
    skills = GithubSkillRegistry(
        "trailofbits/skills-curated",
        ["openai-spreadsheet"],
        path="plugins/openai-spreadsheet/skills",
    )
    agent = Agent(
        name="Analyst",
        description="An assistant that builds spreadsheets.",
        instructions="Help the user with data and spreadsheets.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        skills=skills,
    )
    async with agent:
        response = await agent.run("Create budget.xlsx with a monthly budget for a student.")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
