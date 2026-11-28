"""Any model through OpenRouter, with others to fall back on. Needs OPENROUTER_API_KEY."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient


async def main() -> None:
    client = OpenRouterChatCompletionClient(
        model="openai/gpt-6-luna",
        fallback_models=["anthropic/claude-haiku-5.5"],  # tried in order when the first fails
        app_name="max_ai_examples",
        # Claude uses the provider's cache only when asked; other models ignore this.
        options={"cache_control": {"type": "ephemeral"}},
    )
    agent = Agent(
        name="Assistant",
        description="A simple assistant.",
        instructions="Answer briefly and clearly.",
        client=client,
    )
    try:
        state = await agent.run("What is an AI agent?")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
