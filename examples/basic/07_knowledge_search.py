"""Search local documentation. Requires maxai[embeddings]."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry


async def main() -> None:
    knowledge = LocalKnowledgeRegistry(
        name="framework_docs",
        description="Max AI framework documentation.",
        base_path=Path(__file__).resolve().parents[1] / "local",
    )
    agent = Agent(
        name="DocumentationAssistant",
        description="An assistant that answers using documentation.",
        instructions="Search framework_docs before answering.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        knowledge=[knowledge],
    )
    async with agent:
        response = await agent.run("What does memory do in Max AI?")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
