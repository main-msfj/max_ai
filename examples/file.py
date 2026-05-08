"""Minimal example for serving your own Agent in the Max AI Web UI."""

from max_ai.base.agent import Agent
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.models import ModelConfig
from max_ai.ui import server
from max_ai.executor import DockerExecutor
from docker_tools import (
    add,
    multiply,
    slugify,
    word_count,
    summarize_numbers,
    session_echo,
)

from pathlib import Path

tool_source = Path(__file__).parent / "docker_tools.py"


executor = DockerExecutor(user_id="001", tool_source=tool_source)


client = OllamaChatCompletionClient(
    model="qwen3:4b",
    host="http://ollama:11434",
    config=ModelConfig(
        max_context_window=15000,
        supports_function_calling=True,
        supports_thinking=True,
        supports_vision=False,
    ),
    think=True,
    num_predict=3000,
)

agent = Agent(
    name="Sara",
    description="Example Max AI agent.",
    instructions="You are a helpful assistant. Be concise and useful.",
    client=client,
    toolset=[add, multiply, slugify, word_count, summarize_numbers, session_echo],
    executor=executor,
)

if __name__ == "__main__":
    server(agent, host="0.0.0.0", port=8000)
