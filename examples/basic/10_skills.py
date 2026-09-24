"""Use a skill included in the repository."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.skills.local import LocalSkillRegistry


async def main() -> None:
    skills = LocalSkillRegistry(
        source=Path(__file__).resolve().parents[1] / "LocalSkills",
        skills=["create-report"],
    )
    agent = Agent(
        name="Writer",
        description="An assistant that creates reports.",
        instructions="Use the create-report skill to create reports.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        skills=skills,
    )
    async with agent:
        response = await agent.run("Create a short report about solar energy.")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
