"""A full agent with tools, memory, knowledge, skills, and sessions."""

import asyncio
from pathlib import Path

from max_ai.agents import Agent
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.middleware import LoggingMiddleware
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.capabilities.skills.github import GithubSkillRegistry
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.embeddings import FastEmbedEmbedding
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
LOCAL_DIR = EXAMPLES_DIR / "local"


def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """Calculate the balance with annual compound interest."""
    return round(principal * (1 + annual_rate / 100) ** years, 2)


async def main() -> None:
    client = OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY")
    memory = LocalMemoryRegistry(
        base_path=LOCAL_DIR,
        tool_mode=MemoryToolMode.FULL,
        embedding=FastEmbedEmbedding(),
    )
    knowledge = LocalKnowledgeRegistry(
        name="framework_docs",
        description="Max AI framework documentation.",
        base_path=LOCAL_DIR,
        tool_mode=KnowledgeToolMode.FULL,
        embedding=FastEmbedEmbedding(),
    )
    skills = GithubSkillRegistry("trailofbits/skills-curated", ["openai-spreadsheet"], path="plugins/openai-spreadsheet/skills")
    agent = Agent(
        name="FullAssistant",
        description="An assistant with local framework components.",
        instructions="Help the user. Use the tools, memory, and documentation when helpful.",
        client=client,
        toolset=[
            FunctionAsTool(
                calculate_compound_interest,
                approval_mode=ToolApprovalMode.AUTO_APPROVED,
                read_only=True,
            )
        ],
        memory=memory,
        knowledge=[knowledge],
        skills=skills,
        compaction=SummaryCompaction(threshold=0.8, keep_ratio=0.4),
        middlewares=[LoggingMiddleware()],
    )
    store = LocalSessionStore(LOCAL_DIR / "sessions")
    context = await store.load("user_ana", "full_demo") or RunContext(
        user_id="user_ana", session_id="full_demo"
    )
    async with agent:
        response = await agent.run(
            "How much will I have with 1000 euros at 5% for 3 years?",
            run_context=context,
        )
        await store.save(response.context)
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
