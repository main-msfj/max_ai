"""A full Agent on OpenAI, run inside the MaxAI Textual CLI.

From the repository root:
    .venv/bin/python example-for-deployment/cli_global_agent_openai.py [--session <id>]

Loads OPENAI_API_KEY from the environment/.env. The Agent carries everything
it is (model, tools, memory, knowledge, skills, compaction); the CLI is only
the host: it saves every turn in the store and resumes sessions.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from tools_shared import (
    LOCAL_DIR,
    USER_ID,
    calculate_compound_interest,
    save_report,
    session_arg,
)

from max_ai.agents import Agent
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.middleware import TracingMiddleware, configure_langfuse
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.capabilities.skills.github import GithubSkillRegistry
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.cli import run_cli
from max_ai.core.embeddings import FastEmbedEmbedding
from max_ai.core.model.llm import ModelConfig
from max_ai.types.tools import ToolApprovalMode

MODEL = "gpt-5.6-luna"


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment or in .env.")

    # The client stores the env var's name, never the key itself.
    client = OpenAIChatCompletionClient(
        model=MODEL,
        reasoning_effort="none",
        # Room for a whole file in one tool call; 32K is the framework's cap
        # and compaction reserves it in the window.
        max_tokens=32_000,
        # Default window 128K; MAX_CONTEXT_WINDOW changes it (kept within 128K–1M).
        config=ModelConfig(
            supports_function_calling=True,
            max_context_window=int(os.getenv("MAX_CONTEXT_WINDOW") or 0),
        ),
    )
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None
    # COMPACTION_THRESHOLD=0.05 compacts at ~5% of the window, to watch it happen.
    threshold = float(os.getenv("COMPACTION_THRESHOLD") or 0.8)

    agent = Agent(
        name="LocalDemo",
        description="A conversational agent with local components and an OpenAI model.",
        instructions="Continue the conversation and help the user with their requests.",
        client=client,
        toolset=[
            FunctionAsTool(calculate_compound_interest, approval_mode=ToolApprovalMode.AUTO_APPROVED),
            FunctionAsTool(save_report, approval_mode=ToolApprovalMode.ASK_APPROVED),
        ],
        # Backend only: each run binds it to its RunContext's user and session.
        memory=LocalMemoryRegistry(
            base_path=LOCAL_DIR, tool_mode=MemoryToolMode.FULL,
            embedding=FastEmbedEmbedding(),  # search_memory by meaning, in any language
        ),
        knowledge=[
            LocalKnowledgeRegistry(
                name="framework",
                description="How the max_ai framework works: loop, tools, gates, memory, knowledge, skills.",
                base_path=LOCAL_DIR,
                tool_mode=KnowledgeToolMode.FULL,
            ),
        ],
        skills=GithubSkillRegistry("trailofbits/skills-curated", ["openai-spreadsheet"], path="plugins/openai-spreadsheet/skills"),
        compaction=SummaryCompaction(threshold=threshold, keep_ratio=threshold / 2),
        # With LANGFUSE_PUBLIC_KEY/SECRET_KEY set, every turn is a trace in Langfuse.
        middlewares=[TracingMiddleware()] if tracing else [],
    )

    try:
        async with agent:
            await run_cli(
                agent,
                store=LocalSessionStore(LOCAL_DIR / "sessions"),
                user_id=USER_ID,
                session_id=session_arg(),
            )
    finally:
        await client.client.close()
        if tracing:
            tracing.shutdown()  # sends the spans still buffered


if __name__ == "__main__":
    asyncio.run(main())
