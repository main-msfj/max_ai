"""Define an instruction layer by extending CoreLayer."""

import asyncio

from max_ai.agents import Agent
from max_ai.base.component import Component
from max_ai.base.layer import CoreLayer
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.model.stacks import StackConfig


class EnglishStyleLayer(Component[StackConfig], CoreLayer):
    component_schema = StackConfig
    component_type = "prompts"

    def __init__(self) -> None:
        super().__init__(name="EnglishStyleLayer")

    def _default_template(self) -> str:
        return "Write in plain English and use short sentences."


async def main() -> None:
    agent = Agent(
        name="Assistant",
        description="An assistant with a custom style layer.",
        instructions="Help with general questions.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        prompt_layers=[EnglishStyleLayer()],
    )
    async with agent:
        response = await agent.run("Explain what an API is.")
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
