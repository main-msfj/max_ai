"""One agent, two users at once: the agent keeps nothing of a conversation."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.types.run_context import RunContext


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant for many users.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
    )
    ana, luis = RunContext(user_id="ana"), RunContext(user_id="luis")
    try:
        states = await asyncio.gather(agent.run("Greet Ana.", ana), agent.run("Greet Luis.", luis))
        for ctx, state in zip((ana, luis), states, strict=True):
            print(ctx.user_id, "->", state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
