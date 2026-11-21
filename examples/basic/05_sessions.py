"""Save a conversation and continue it in a later run. Run it twice."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.types.run_context import RunContext


async def main() -> None:
    store = LocalSessionStore(Path("./local/sessions"))
    # The saved conversation, or a new one the first time.
    ctx = await store.load("ana", "demo") or RunContext(user_id="ana", session_id="demo")
    agent = Agent(
        name="Assistant",
        description="An assistant with conversation history.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
    )
    try:
        task = "What did I tell you before?" if ctx.messages else "Remember this: my dog is called Luna."
        state = await agent.run(task, ctx)
        print(state.text)
        await store.save(ctx)  # the conversation stays in ctx, not in the answer
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
