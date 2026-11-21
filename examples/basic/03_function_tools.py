"""A Python function as a tool: its name, docstring and types are what the model reads."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.tools import ToolApprovalMode


def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate the balance with annual compound interest. ``annual_rate`` is a percentage: 5 for 5%."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


async def main() -> None:
    agent = Agent(
        name="FinancialAdvisor",
        description="An assistant that calculates compound interest.",
        instructions="Use calculate_compound_interest to calculate the final balance.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        # It only reads, so it runs without asking the user.
        toolset=[FunctionAsTool(calculate_compound_interest, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
    )
    try:
        state = await agent.run("How much will I have after investing 1000 euros at 5% for 3 years?")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
