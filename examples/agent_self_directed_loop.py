"""Serve an agent that uses the self-directed ReAct loop, on Ollama, in the Web UI.

``ReActLoopSelfDirected`` lets the *model* manage its own plan via the
``update_plan`` tool (planning-as-tool), rather than being steered by the
loop the way ``ReActLoopPlanning`` does. The loop auto-registers the
``update_plan`` tool for each run, so you only have to pass the loop in.

The plan the model builds is streamed to the Web UI as ``PlanningEvent``s
and rendered in the live plan panel above the chat (and in the trace panel
on the right).

Run:
    python examples/agent_self_directed_loop.py
Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import os
import typing as t
from pathlib import Path

from dotenv import load_dotenv

from max_ai.ui import create_app
from max_ai.base.agent import Agent
from max_ai.base.tools import CoreTool
from max_ai.core.models import ModelConfig
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.clients.openai import OpenAIChatCompletionClient
from max_ai.middleware import ConsoleTraceMiddleware
from max_ai.reasoning.react_self_directed import ReActLoopSelfDirected
from max_ai.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    StdioMCPServerConfig,
    create_mcp_tools,
)


EXAMPLE_DIR = Path(__file__).parent
USER_ID = "user1234"
SESSION_ID = "session_self_directed"

load_dotenv()


def build_client_ollama() -> OllamaChatCompletionClient:
    """Local Ollama client. Adjust ``model``/``host`` to your setup."""
    return OllamaChatCompletionClient(
        model="qwen3.5:4b-q4_K_M",
        host="http://ollama:11434",
        config=ModelConfig(
            max_context_window=12000,
            supports_function_calling=True,  # required: the model calls update_plan
            supports_thinking=True,
        ),
        max_tokens=3000,
    )


def build_mcp_server_configs() -> list[HTTPServerConfig | StdioMCPServerConfig]:
    """Build MCP server configs for internet access (Tavily web search).

    Reads TAVILY_URL (and optional TAVILY_KEY) from the environment / .env.
    If TAVILY_URL is unset, no MCP servers are configured and the agent runs
    without internet tools.
    """
    configs: list[HTTPServerConfig | StdioMCPServerConfig] = []

    tavily_url = os.getenv("TAVILY_URL")
    if tavily_url:
        configs.append(
            HTTPServerConfig(
                server_id="tavily-websearch",
                url=tavily_url,
                token=os.getenv("TAVILY_KEY"),
            )
        )

    return configs

def build_client_openai() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="gpt-5.4-nano",  # gpt-5-nano gpt-4.1-nano
        api_key=os.getenv("OPENAI_KEY"),
        config=ModelConfig(
            max_context_window=15000,
            supports_function_calling=True,
            supports_vision=True,
        ),
        max_tokens=3000,
    )



async def build_mcp_client_manager() -> tuple[
    MCPClientManager,
    list[CoreTool | t.Callable[..., t.Any]],
]:
    """Connect to the MCP servers and return the manager plus the tools."""
    return await create_mcp_tools(build_mcp_server_configs())


def build_agent(
    mcp_tools: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
) -> Agent:
    """An agent whose reasoning loop is the self-directed (planning-as-tool) loop.

    Passing ``reasoning=ReActLoopSelfDirected()`` is the only thing that makes
    this a self-directed agent — the loop registers its own ``update_plan``
    tool per run, so nothing else needs wiring. ``mcp_tools`` (e.g. Tavily web
    search) give it internet access.
    """
    return Agent(
        name="Pathfinder",
        description="Agent that plans its own work via the update_plan tool.",
        instructions=(
            "You are a helpful assistant with internet access. For any "
            "multi-step task, call the update_plan tool first to lay out your "
            "plan, then work through it, using web search when you need facts, "
            "and calling update_plan again to mark steps done as you go."
        ),
        client=build_client_openai(),
        reasoning=ReActLoopSelfDirected(max_loop_iterations=12),
        toolset=list(mcp_tools or []),
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
