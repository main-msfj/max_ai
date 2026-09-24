"""Ask for approval before saving a report to a file."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.tools import ToolApprovalMode


def save_report(report: str) -> str:
    """Save a report to a local file."""
    path = Path("./local/reports/report.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return f"Informe guardado en {path}"


async def main() -> None:
    agent = Agent(
        name="ReportWriter",
        description="An assistant that saves reports to a local file.",
        instructions="Use save_report after writing a report.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        toolset=[FunctionAsTool(save_report, approval_mode=ToolApprovalMode.ASK_APPROVED)],
    )
    async with agent:
        response = await agent.run("Write a short report about solar energy and save it.")
        for pending in response.pending_approvals:
            print(pending.tool_name, pending.parameters)
            if input("Approve saving the report? (y/n): ").lower() == "y":
                pending.approve()
            else:
                pending.reject("The user did not approve saving the file.")
        if response.pending_approvals:
            response = await agent.resume(response.context)
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
