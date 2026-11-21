"""The smallest agent: one question, one answer. Needs OPENAI_API_KEY."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="A simple assistant.",
        instructions="Answer briefly and clearly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),  # the key comes from the environment
    )
    try:
        state = await agent.run("Explain an AI agent in one sentence.")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
