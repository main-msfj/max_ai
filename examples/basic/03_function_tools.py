"""Give the agent a compound interest calculator tool."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.tools import ToolApprovalMode


def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate the balance with annual compound interest."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


async def main() -> None:
    agent = Agent(
        name="FinancialAdvisor",
        description="An assistant that calculates compound interest.",
        instructions="Use calculate_compound_interest to calculate the final balance.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        toolset=[FunctionAsTool(calculate_compound_interest, approval_mode=ToolApprovalMode.AUTO_APPROVED)],
    )
    async with agent:
        response = await agent.run("How much will I have after investing 1000 euros at 5% for 3 years?")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
