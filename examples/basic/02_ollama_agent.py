"""Use a local model. Ollama and a downloaded model are required."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.ollama import OllamaChatCompletionClient


async def main() -> None:
    client = OllamaChatCompletionClient(model="llama3.1:8b")
    agent = Agent(
        name="Assistant",
        description="A local assistant.",
        instructions="Answer briefly in English.",
        client=client,
    )
    async with agent:
        response = await agent.run("What is an AI agent?")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
