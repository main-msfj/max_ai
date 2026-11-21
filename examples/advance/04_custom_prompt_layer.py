"""A prompt layer of your own: a part of the system prompt with its own text."""

import asyncio

from max_ai.agents import Agent
from max_ai.base.component import Component
from max_ai.base.layer import CoreLayer
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.model.stacks import StackConfig


class PlainStyleLayer(Component[StackConfig], CoreLayer):
    component_schema = StackConfig
    component_type = "prompts"

    def __init__(self) -> None:
        super().__init__(name="PlainStyleLayer")

    def _default_template(self) -> str:
        return "Write in plain English and use short sentences."


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant with a style layer of its own.",
        instructions="Help with general questions.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        prompt_layers=[PlainStyleLayer()],  # added after the framework's own layers
    )
    try:
        state = await agent.run("Explain what an API is.")
        print(state.text)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
