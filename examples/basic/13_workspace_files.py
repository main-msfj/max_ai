"""Create a file in a user's workspace."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.types.run_context import RunContext


async def main() -> None:
    workspace = LocalWorkspace(root=Path("./local/workspaces"))
    folder = workspace.materialize("ana").workspace_dir
    (folder / "note.txt").write_text("My favorite color is blue.", encoding="utf-8")
    agent = Agent(
        name="Assistant",
        description="An assistant with workspace files.",
        instructions="Read note.txt from the workspace before answering.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        workspace=workspace,
    )
    async with agent:
        response = await agent.run("What is my favorite color?", run_context=RunContext(user_id="user_ana"))
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
