"""Log every step, and send traces when an OpenTelemetry provider is configured."""

import asyncio
import logging

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.middleware import LoggingMiddleware, TracingMiddleware


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agent = Agent(
        name="Assistant",
        description="An observable assistant.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        # In order: the first one is the outermost around each turn, model call and tool.
        middlewares=[LoggingMiddleware(), TracingMiddleware()],
    )
    try:
        state = await agent.run("Greet me in one sentence.")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
