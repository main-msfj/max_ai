"""A sandbox: where the agent's tools run and its files live. With one, the
harness adds the file tools; BashTool is yours to give."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.bash import BashTool
from max_ai.types.run_context import RunContext

ROOT = Path("./local/box")  # .agent/ is created inside: workspace, skills, memory


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sandbox = LocalSandbox(root=str(ROOT))  # this computer: easy to look at, not isolation
    ctx = RunContext(user_id="ana", session_id="demo")

    # A file the user attached, put in the folder of their session.
    box = sandbox.bind(ctx.user_id)
    await box.prepare()
    await box.upload_file("demo", "note.txt", b"My favorite color is blue.")

    agent = Agent(
        name="Assistant",
        description="An assistant that works with files.",
        instructions="Read the files in your working directory before you answer.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        toolset=[BashTool()],
        sandbox=sandbox,
    )
    try:
        state = await agent.run("What is my favorite color? It is in note.txt.", ctx)
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
