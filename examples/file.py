"""Complete example for serving a Docker-backed Agent in the Max AI Web UI."""

from __future__ import annotations

import asyncio
import os
import typing as t
from pathlib import Path

from dotenv import load_dotenv

from max_ai.ui import create_app
from max_ai.base.agent import Agent
from max_ai.base.tools import CoreTool
from max_ai.executor import DockerExecutor
from max_ai.core.models import ModelConfig
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.clients.openai import OpenAIChatCompletionClient
from max_ai.middleware import ConsoleTraceMiddleware
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.memory import SQLiteMemoryRegistry
from max_ai.capabilities.context import SQLiteContextRegistry
from max_ai.capabilities.routines import SQLiteRoutineRegistry
from max_ai.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    StdioMCPServerConfig,
    create_mcp_tools,
)
from docker_tools import get_weather, check_link


EXAMPLE_DIR = Path(__file__).parent
TOOL_SOURCE = EXAMPLE_DIR / "docker_tools.py"
SKILLS_SOURCE = EXAMPLE_DIR / "LocalSkills"
SKILL_NAMES = ["create-report", "create-ppt"]
TOOLSET = [get_weather, check_link]
USER_ID = "user123"
SESSION_ID = "session1234"
BACKEND_DIR = EXAMPLE_DIR / "backend-local"

load_dotenv()


def build_client_ollama() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model="qwen3.5:4b-q4_K_M",  # "qwen3.5:4b-q4_K_M" #"gemma4:e2b-it-q4_K_M", qwen3.5:9b-q4_K_M
        host="http://ollama:11434",
        config=ModelConfig(
            max_context_window=12000,
            supports_function_calling=True,
            supports_thinking=True,
            supports_vision=True,
        ),
        # think=True,
        max_tokens=3000,
    )


def build_client_openai() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="gpt-5.4-nano",  # gpt-5-nano gpt-4.1-nano
        api_key=os.getenv("OPENAI_KEY"),
        config=ModelConfig(
            max_context_window=15000,
            supports_function_calling=True,
            # supports_vision=True,
        ),
        max_tokens=3000,
    )


def build_executor() -> DockerExecutor:
    return DockerExecutor(image="maxai-sandbox:py311")


def build_skills() -> LocalSkillRegistry:
    return LocalSkillRegistry(
        source=SKILLS_SOURCE,
        skills=SKILL_NAMES,
    )


def build_memory() -> SQLiteMemoryRegistry:
    return SQLiteMemoryRegistry(
        user_id=USER_ID,
        base_path=EXAMPLE_DIR,
    )


def build_context() -> SQLiteContextRegistry:
    return SQLiteContextRegistry(
        user_id=USER_ID,
        session_id=SESSION_ID,
        base_path=EXAMPLE_DIR,
    )


def build_routines() -> SQLiteRoutineRegistry:
    return SQLiteRoutineRegistry(base_path=EXAMPLE_DIR)


def build_mcp_server_configs() -> list[HTTPServerConfig | StdioMCPServerConfig]:
    """Build optional MCP server configs from environment variables."""
    configs: list[HTTPServerConfig | StdioMCPServerConfig] = []

    tavily_url = os.getenv("TAVILY_URL")
    if tavily_url:
        configs.append(
            HTTPServerConfig(
                server_id="tavily-websearch",
                url=tavily_url,
                token=os.getenv("TAVILY_KEY")
            )
        )

    return configs


async def build_mcp_client_manager() -> tuple[
    MCPClientManager,
    list[CoreTool | t.Callable[..., t.Any]],
]:
    return await create_mcp_tools(build_mcp_server_configs())


def build_agent(
    mcp_tools: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
) -> Agent:
    return Agent(
        name="Sara",
        description="Example Max AI agent.",
        instructions=("You are a helpful assistant. Be concise and useful. "),
        client=build_client_ollama(),
        memory=build_memory(),
        # routines=build_routines(),
        logbook=build_context(),
        skills=build_skills(),
        toolset=[*TOOLSET, *(mcp_tools or [])],
        executor=build_executor(),
        middlewares=[ConsoleTraceMiddleware()],
    )


async def main() -> None:
    import uvicorn

    mcp_manager, mcp_tools = await build_mcp_client_manager()
    agent = build_agent(mcp_tools)
    app = create_app(agent, user_id=USER_ID, session_id=SESSION_ID)

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    web_server = uvicorn.Server(config)

    try:
        await web_server.serve()
    finally:
        await mcp_manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
