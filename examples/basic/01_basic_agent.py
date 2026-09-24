"""Minimal OpenAI agent."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-5.6-luna",
        api_key="YOUR_API_KEY",
    )
    agent = Agent(
        name="Assistant",
        description="A simple assistant.",
        instructions="Answer briefly and clearly.",
        client=client,
    )

    async with agent:
        response = await agent.run("Explain an AI agent in one sentence.")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
