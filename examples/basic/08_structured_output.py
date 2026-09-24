"""Return a response validated with a Pydantic model."""

import asyncio

from pydantic import BaseModel

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


class City(BaseModel):
    name: str
    country: str


async def main() -> None:
    agent = Agent(
        name="Geographer",
        description="Returns structured data.",
        instructions="Answer the question with city details.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        output_format=City,
    )
    async with agent:
        response = await agent.run("Give me the capital city of Peru.")
        print(response.final_message.structured_output)


if __name__ == "__main__":
    asyncio.run(main())
