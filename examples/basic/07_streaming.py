"""Print the answer as the model writes it."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant that streams its answer.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
    )
    try:
        # Only the text; run_stream_events gives every event (tools, usage, approvals).
        async for text in agent.run_stream("Write three ideas for learning Python."):
            print(text, end="", flush=True)
        print()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
