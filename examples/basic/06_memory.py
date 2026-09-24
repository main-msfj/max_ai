"""Save a user preference in local memory."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.types.run_context import RunContext


async def main() -> None:
    memory = LocalMemoryRegistry(base_path=Path("./local"))
    await memory.bind("user_ana", "first").create_or_update("language", "Prefers English")
    agent = Agent(
        name="Assistant",
        description="An assistant with memory.",
        instructions="Use memory to adapt your answers.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        memory=memory,
    )
    async with agent:
        response = await agent.run(
            "What is my preferred language?",
            run_context=RunContext(user_id="user_ana", session_id="first"),
        )
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
