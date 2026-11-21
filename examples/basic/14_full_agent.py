"""Most pieces together: a tool, a sandbox, memory, a summary when the
conversation grows, logging and a saved session."""

import asyncio
import logging
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.loop.react import ReactLoop
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.middleware import LoggingMiddleware
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

LOCAL = Path("./local")


def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate the balance with annual compound interest. ``annual_rate`` is a percentage: 5 for 5%."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    (LOCAL / "box").mkdir(parents=True, exist_ok=True)
    interest = FunctionAsTool(
        calculate_compound_interest, approval_mode=ToolApprovalMode.AUTO_APPROVED, read_only=True
    )
    agent = Agent(
        name="FullAssistant",
        description="An assistant with most of the framework's pieces.",
        instructions="Help the user. Use your tools and what you remember of them.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        toolset=[interest],
        memory=Memory(),
        sandbox=LocalSandbox(root=str(LOCAL / "box")),
        # When the conversation fills the window, the old part becomes a summary.
        reasoning=ReactLoop(compaction=SummaryCompaction()),
        middlewares=[LoggingMiddleware()],
    )
    store = LocalSessionStore(LOCAL / "sessions")
    ctx = await store.load("ana", "full") or RunContext(user_id="ana", session_id="full")
    try:
        state = await agent.run("How much will I have with 1000 euros at 5% for 3 years?", ctx)
        print(state.text)
        await store.save(ctx)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
