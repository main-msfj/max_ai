"""Continuous chat with the new Agent stack: two demo tools, one of each
approval mode, and a terminal y/n prompt whenever a tool needs approval.

From the repository root:
    .venv/bin/python -m examples.agent_local_openai

Loads OPENAI_KEY (or OPENAI_API_KEY / OPENA_KEY) from the environment/.env.
Local execution is not sandboxed. Pending approvals are never auto-approved.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents import Agent
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.clients.openai import OpenAIChatCompletionClient
from max_ai.core.event_type import ToolApprovalEvent, ToolCallEvent, ToolCallResponseEvent
from max_ai.core.models import ModelConfig
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


def get_weather(city: str) -> dict:
    """Return the current weather for a city. Read-only, no side effects."""
    return {"city": city, "condition": "soleado", "temp_c": 24}


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to someone. Has a real external side effect."""
    print(f"[send_email] (simulado) para={to!r} asunto={subject!r} cuerpo={body!r}")
    return {"sent": True, "to": to}


async def _drive(agent: Agent, stream) -> AgentResponse:
    """Print tool activity as it streams and return the terminal AgentResponse."""
    response: AgentResponse | None = None
    reasons: dict[str, str] = {}
    async for event in stream:
        if isinstance(event, ToolCallEvent):
            description = event.parameters.get("description")
            print(f"Tool: {event.tool_name} — {description}" if description
                  else f"Tool: {event.tool_name} {event.parameters}")
        elif isinstance(event, ToolApprovalEvent):
            reasons[event.tool_call_id] = event.reason_for_approval or ""
        elif isinstance(event, ToolCallResponseEvent):
            print(f"Resultado: {'OK' if event.tool_result.success else 'error'}")
        elif isinstance(event, AgentResponse):
            response = event
    assert response is not None

    while response.needs_approval or response.needs_input:
        for record in response.pending_approvals:
            reason = reasons.get(record.id) or f"{record.tool_name}({record.parameters})"
            answer = input(f"¿Autorizas esto? {reason} [y/n] ").strip().lower()
            if answer == "y":
                record.approve()
            else:
                record.reject()
        for record in response.pending_questions:
            answer = input(f"{record.input_question} ")
            response.context.tool_state.apply_user_answer(record.id, answer)
        response = await _drive(agent, agent.resume_stream_events(response.context))

    return response


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENA_KEY")
    if not key:
        raise SystemExit("Configura OPENAI_KEY en el entorno o en .env.")

    client = OpenAIChatCompletionClient(
        model="gpt-5.6-luna",
        api_key=key,
        reasoning_effort="none",
        max_tokens=1500,
        config=ModelConfig(supports_function_calling=True),
    )
    toolset = [
        BashTool(),
        FunctionAsTool(get_weather, approval_mode=ToolApprovalMode.AUTO_APPROVED),
        FunctionAsTool(send_email, approval_mode=ToolApprovalMode.ASK_APPROVED),
    ]

    ctx: RunContext | None = None
    try:
        async with Agent(
            name="LocalDemo",
            description="Agente conversacional con componentes locales y modelo OpenAI.",
            instructions="Continua la conversacion con el usuario y ayudale con todo lo que necesite.",
            client=client,
            toolset=toolset,
        ) as agent:
            print("Escribe tu mensaje (Ctrl+C para salir).")
            while True:
                try:
                    task = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not task:
                    continue
                response = await _drive(agent, agent.run_stream_events(task, run_context=ctx))
                ctx = response.context
                if response.final_message is not None:
                    print(response.final_message.text())
    finally:
        await client.client.close()


if __name__ == "__main__":
    asyncio.run(main())
