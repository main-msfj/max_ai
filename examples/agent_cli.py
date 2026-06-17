"""Chat with a max_ai agent from the terminal (Rich CLI).

Same agent wiring as ``agent_self_directed_loop.py`` — a self-directed
ReAct agent with optional Tavily web search — but instead of serving the
web UI it opens an interactive REPL in your terminal via ``max_ai.cli``.

You get live streamed answers, the agent's plan, its tool calls, a thinking
spinner, and inline human-input: when the agent calls its native
``structure_human_in_loop`` tool, the CLI prompts you right in the chat.

Run:
    python examples/agent_cli.py
Then just start typing. ``/exit`` or Ctrl-D to quit.
"""

from __future__ import annotations

import asyncio
import os
import typing as t

from dotenv import load_dotenv

from max_ai.cli import run_repl
from max_ai.base.agent import Agent
from max_ai.base.tools import CoreTool
from max_ai.core.models import ModelConfig
from max_ai.clients.openai import OpenAIChatCompletionClient
from max_ai.reasoning.react_self_directed import ReActLoopSelfDirected
from max_ai.mcp import (
    HTTPServerConfig,
    MCPClientManager,
    StdioMCPServerConfig,
    create_mcp_tools,
)


load_dotenv()


def build_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="gpt-5.4-nano",
        api_key=os.getenv("OPENAI_KEY"),
        config=ModelConfig(
            max_context_window=15000,
            supports_function_calling=True,
        ),
        max_tokens=3000,
    )


def build_mcp_server_configs() -> list[HTTPServerConfig | StdioMCPServerConfig]:
    """Tavily web search if TAVILY_URL is set; otherwise no internet tools."""
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


async def build_mcp_client_manager() -> tuple[
    MCPClientManager,
    list[CoreTool | t.Callable[..., t.Any]],
]:
    return await create_mcp_tools(build_mcp_server_configs())


def build_agent(
    mcp_tools: t.Sequence[CoreTool | t.Callable[..., t.Any]] | None = None,
) -> Agent:
    return Agent(
        name="Pathfinder",
        description="Agent that plans its own work and asks when it needs to.",
        instructions=(
            "You are a helpful assistant"
        ),
        client=build_client(),
        reasoning=ReActLoopSelfDirected(max_loop_iterations=12),
        toolset=list(mcp_tools or []),
    )


async def main() -> None:
    mcp_manager, mcp_tools = await build_mcp_client_manager()
    agent = build_agent(mcp_tools)
    try:
        await run_repl(agent)
    finally:
        await mcp_manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
