"""An answer that follows a Pydantic model."""

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
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        output_format=City,
    )
    try:
        state = await agent.run("Give me the capital city of Peru.")
        # A City, already validated; None when the turn stopped before an answer.
        print(state.message.structured_output if state.message else state.finish_reason)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
