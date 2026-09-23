"""Shared demo Agent for the numbered examples (01_, 02_, ...).

``run_agent_in_cli(client, provider)`` builds the same tools, memory,
knowledge and skills for any client and opens the MaxAI Textual CLI.
Each numbered example only builds its client. ``max_ai.cli`` owns the
terminal interaction, streaming, approvals, questions, and session context.
"""

from __future__ import annotations

import os

from pathlib import Path

from max_ai.capabilities.compaction import SummaryCompaction
from max_ai.agents import Agent
from max_ai.base.clients import CoreChatCompletionClient
from max_ai.base.knowledge import KnowledgeToolMode
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.cli import run_repl
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode

EXAMPLES_DIR = Path(__file__).resolve().parent
LOCAL_DIR = EXAMPLES_DIR / "local"
USER_ID = "user_001"
SESSION_ID = "demo"


def get_weather(city: str) -> dict[str, str | int]:
    """Return the current weather for a city. Read-only, no side effects."""
    return {"city": city, "condition": "soleado", "temp_c": 24}


def send_email(to: str, subject: str, body: str) -> dict[str, str | bool]:
    """Send an email to someone. Has a real external side effect."""
    print(f"[send_email] (simulado) para={to!r} asunto={subject!r} cuerpo={body!r}")
    return {"sent": True, "to": to}


async def run_agent_in_cli(client: CoreChatCompletionClient, provider: str) -> None:
    """Same tools, memory, knowledge and skills for any client."""
    toolset = [
        BashTool(),
        FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED),
        FunctionAsTool(send_email, approval_mode=ToolApprovalMode.ASK_APPROVED),
    ]

    # Fixture data under examples/local/ — see that folder for the raw files.
    memory = LocalMemoryRegistry(
        user_id=USER_ID,
        session_id=SESSION_ID,
        base_path=LOCAL_DIR,
        tool_mode=MemoryToolMode.FULL,
    )
    knowledge = [
        LocalKnowledgeRegistry(
            name="framework",
            description=(
                "Facts about how the max_ai framework itself works: the agent "
                "loop, tools/approval, completion gates, loop guards, memory, "
                "knowledge and skills."
            ),
            base_path=LOCAL_DIR,
            tool_mode=KnowledgeToolMode.FULL,
        ),
    ]
    skills = LocalSkillRegistry(
        source=EXAMPLES_DIR / "LocalSkills",
        skills=["create-report", "create-ppt"],
    )

    try:
        async with Agent(
            name="LocalDemo",
            description=f"Agente conversacional con componentes locales y modelo {provider}.",
            instructions="Continua la conversacion con el usuario y ayudale con todo lo que necesite.",
            client=client,
            toolset=toolset,
            memory=memory,
            knowledge=knowledge,
            skills=skills,
            # COMPACTION_THRESHOLD=0.05 compacts at ~5% of the message room,
            # to watch it happen in a few turns (default 0.8).
            compaction=SummaryCompaction(
                threshold=float(os.getenv("COMPACTION_THRESHOLD") or 0.8),
            ),
        ) as agent:
            await run_repl(
                agent,
                show_thinking=True,
                initial_context=RunContext(user_id=USER_ID, session_id=SESSION_ID),
            )
    finally:
        await client.client.close()
