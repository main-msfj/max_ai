"""A middleware: code around the turn, the model call and each tool call.
Override only the places you need."""

import asyncio
from collections.abc import AsyncGenerator

from max_ai.agents import Agent
from max_ai.base.middleware import CoreMiddleware, ModelCall, NextModel, NextTool, NextTurn
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.event_type import CoreEvent
from max_ai.types.completions import LLMResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult


class ConsoleMiddleware(CoreMiddleware):
    async def turn(self, ctx: RunContext, call_next: NextTurn) -> AsyncGenerator[CoreEvent]:
        print(f"turn of {ctx.user_id} starts")
        async for event in call_next():
            yield event
        print("turn ends")

    async def model(self, ctx: RunContext, call: ModelCall, call_next: NextModel) -> LLMResponse:
        response = await call_next(call)
        print(f"model answered: {response.usage.tokens_output} tokens")
        return response

    async def tool(
        self, ctx: RunContext, record: ToolCallRecord, call_next: NextTool
    ) -> ToolResult | None:
        print(f"tool {record.tool_name} called")
        return await call_next(record)


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant with a middleware of its own.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        middlewares=[ConsoleMiddleware()],
    )
    try:
        state = await agent.run("Say hello briefly.", RunContext(user_id="ana"))
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
