"""Save a conversation and continue it in a later run."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.types.run_context import RunContext


async def main() -> None:
    store = LocalSessionStore(Path("./local/sessions"))
    context = await store.load("user_ana", "demo") or RunContext(user_id="user_ana", session_id="demo")
    agent = Agent(
        name="Assistant",
        description="An assistant with conversation history.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
    )
    async with agent:
        response = await agent.run("Continue our conversation.", run_context=context)
        print(response.final_text)
        await store.save(response.context)


if __name__ == "__main__":
    asyncio.run(main())
