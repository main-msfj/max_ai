"""A tool that waits for the user: the turn stops, you decide, the turn goes on."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


def save_report(report: str) -> str:
    """Save a report to a local file."""
    path = Path("./local/reports/report.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return f"Report saved to {path}"


async def main() -> None:
    agent = Agent(
        name="ReportWriter",
        description="An assistant that saves reports to a local file.",
        instructions="Use save_report after writing a report.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        toolset=[FunctionAsTool(save_report, approval_mode=ToolApprovalMode.ASK_APPROVED)],
    )
    ctx = RunContext(user_id="ana")  # the conversation, and the calls that wait, live here
    try:
        state = await agent.run("Write a short report about solar energy and save it.", ctx)
        if state.finish_reason == "waiting_for_user":
            # 1. Decide each call that waits.
            for record in ctx.tool_state.records.values():
                if not record.is_pending_approval:
                    continue
                print(record.tool_name, record.parameters)
                if input("Approve saving the report? (y/n): ").lower() == "y":
                    await agent.approve(ctx, record.id)
                else:
                    record.reject("The user did not approve saving the file.")
            # 2. Go on with the same conversation: no new task.
            state = await agent.run(None, ctx)
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
