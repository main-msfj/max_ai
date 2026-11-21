"""A tool written as a class: its own schema, and a result built by hand."""

import asyncio
import typing as t

from max_ai.agents import Agent
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class CompoundInterestTool(CoreTool):
    def __init__(self) -> None:
        super().__init__(
            name="calculate_compound_interest",
            description="Calculate a balance with annual compound interest.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,  # read-only calls of one reply may run at once
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "annual_rate": {"type": "number"},
                "years": {"type": "integer"},
            },
            "required": ["principal", "annual_rate", "years"],
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        given = tool_request.parameters
        balance = given["principal"] * (1 + given["annual_rate"] / 100) ** given["years"]
        return ToolResult.success_result(tool_request.id, round(balance, 2))


async def main() -> None:
    agent = Agent(
        name="FinancialAdvisor",
        description="Calculates compound interest.",
        instructions="Use calculate_compound_interest for investment calculations.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        toolset=[CompoundInterestTool()],
    )
    try:
        state = await agent.run("How much will I have with 1000 euros at 5% for 3 years?")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
