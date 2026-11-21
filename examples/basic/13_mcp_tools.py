"""Tools from an MCP server. Its key stays in the environment: the config
keeps only the name of the variable. Needs EXA_API_KEY."""

import asyncio
import os

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.mcp import HTTPSecret, HTTPServerConfig
from max_ai.types.tools import ToolApprovalMode


async def main() -> None:
    agent = Agent(
        name="Researcher",
        description="An assistant that searches the web.",
        instructions="Search the web before you answer, and name your sources.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        mcp=[
            HTTPServerConfig(
                server_id="exa",
                url="https://mcp.exa.ai/mcp",
                secret=HTTPSecret(
                    type="api_key", header="x-api-key",
                    keys={"EXA_API_KEY": os.getenv("EXA_API_KEY")},
                ),
                approval_mode=ToolApprovalMode.AUTO_APPROVED,  # its tools only read
            )
        ],
    )
    try:
        state = await agent.run("What is the Model Context Protocol? One paragraph.")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
