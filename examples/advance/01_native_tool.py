"""Create a native tool by extending CoreTool."""

import asyncio
import typing as t

from max_ai.agents import Agent
from max_ai.base.tools import CoreTool, ToolContext
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.termination import CancellationToken
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class CompoundInterestTool(CoreTool):
    def __init__(self) -> None:
        super().__init__(
            name="calculate_compound_interest",
            description="Calculate a balance with annual compound interest.",
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
            read_only=True,
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

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        parameters = tool_request.parameters
        balance = (
            parameters["principal"]
            * (1 + parameters["annual_rate"] / 100) ** parameters["years"]
        )
        return ToolResult.success_result(tool_request.id, round(balance, 2))


async def main() -> None:
    agent = Agent(
        name="FinancialAdvisor",
        description="Calculate compound interest.",
        instructions="Use calculate_compound_interest for investment calculations.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        toolset=[CompoundInterestTool()],
    )
    async with agent:
        response = await agent.run(
            "How much will I have with 1000 euros at 5% for 3 years?"
        )
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
