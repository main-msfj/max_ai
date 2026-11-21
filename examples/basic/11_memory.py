"""Memory: facts about the user, kept as files in their sandbox and shown to
the model in every new session. Run it twice."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.types.run_context import RunContext

ROOT = Path("./local/box")


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    agent = Agent(
        name="Assistant",
        description="An assistant that remembers its user.",
        instructions="Keep what the user tells you about themselves, and use it.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        memory=Memory(),  # the harness adds save_memory; the files live in the sandbox
        sandbox=LocalSandbox(root=str(ROOT)),
    )
    try:
        # A new session each time: what is remembered comes from the memory, not the history.
        first = await agent.run("I prefer short answers, in Spanish.", RunContext(user_id="ana"))
        print(first.text)
        later = await agent.run("How do I like my answers?", RunContext(user_id="ana"))
        print(later.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
