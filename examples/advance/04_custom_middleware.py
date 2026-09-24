"""Create middleware that observes the start and end of each run."""

import asyncio

from max_ai.agents import Agent
from max_ai.base.middleware import CoreMiddleware, MiddlewareContext
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext


class ConsoleMiddleware(CoreMiddleware):
    async def on_run_start(self, mw: MiddlewareContext, task: list | None) -> None:
        print(f"Starting run for {mw.agent}")

    async def on_run_end(self, mw: MiddlewareContext, response: AgentResponse) -> None:
        print(f"Finished with: {response.finish_reason}")


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant with custom middleware.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        middlewares=[ConsoleMiddleware()],
    )
    async with agent:
        response = await agent.run("Say hello briefly.", run_context=RunContext())
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
