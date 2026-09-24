"""Print the response as its chunks arrive."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant that streams its response.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
    )
    async with agent:
        async for token in agent.run_stream("Write three ideas for learning Python."):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
