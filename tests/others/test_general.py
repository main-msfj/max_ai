
from max_ai.base.agent import Agent
from max_ai.core.models import ModelConfig
from max_ai.types.tools import ToolApprovalMode
from max_ai.middleware.logging import LoggingMiddleware
from max_ai.tools.function_as_tool import FunctionAsTool
from max_ai.clients.ollama.client import OllamaChatCompletionClient

import os 
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

def build_client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=MODEL,
        host=OLLAMA_HOST,
        config=ModelConfig(
            supports_function_calling=True,
            supports_thinking=False,
        ),
        think=False,
    )

def _make_weather_tool() -> FunctionAsTool:
    def get_weather(city: str) -> str:
        """Return the current weather in the given city."""
        return f"Sunny, 22°C in {city}"

    return FunctionAsTool(
        func=get_weather,
        name="get_weather",
        description="Get the current weather for a city.",
        approval_mode=ToolApprovalMode.AUTO_APPROVED,
    )


async def test_agent_run():
    agent = Agent(
        name="test",
        description="testing agent",
        instructions="you are a agent for testing",
        client=build_client(),
        toolset=[_make_weather_tool()],
        middlewares=[LoggingMiddleware(level="info")],
    )

    response = await agent.run(task="Please tell me a joke about fish and check how is the weather in Tokyo and in Nicargua")
    print("Final response:", response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_run())