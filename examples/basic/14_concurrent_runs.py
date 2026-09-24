"""Serve two users at once with the same agent."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.types.run_context import RunContext


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant for multiple users.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
    )
    async with agent:
        responses = await asyncio.gather(
            agent.run("Greet Ana.", run_context=RunContext(user_id="user_ana")),
            agent.run("Greet Luis.", run_context=RunContext(user_id="user_luis")),
        )
        for response in responses:
            print(response.context.user_id, response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
