"""Connect tools from an HTTP MCP server."""

import asyncio

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.mcp import HTTPServerConfig
from max_ai.types.tools import ToolApprovalMode


async def main() -> None:
    agent = Agent(
        name="MCPAssistant",
        description="An assistant with external tools.",
        instructions="Use MCP tools when helpful.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        mcp=[
            HTTPServerConfig(
                server_id="example_server",
                url="http://localhost:8000/mcp",
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
            )
        ],
    )
    async with agent:
        response = await agent.run("Which tools are available?")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
