"""Log calls and send traces when an OpenTelemetry provider is configured."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.middleware import LoggingMiddleware, TracingMiddleware


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An observable assistant.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        middlewares=[LoggingMiddleware(), TracingMiddleware()],
    )
    async with agent:
        response = await agent.run("Greet me in one sentence.")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
