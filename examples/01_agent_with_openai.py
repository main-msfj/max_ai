"""Run the demo Agent on OpenAI inside the MaxAI Textual CLI.

From the repository root:
    .venv/bin/python -m examples.01_agent_with_openai

Loads OPENAI_KEY (or OPENAI_API_KEY / OPENA_KEY) from the environment/.env.
Tools, memory, knowledge and skills come from ``examples/cli_agent.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from examples.cli_agent import run_agent_in_cli
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.model.llm import ModelConfig


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = (
        os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENA_KEY")
    )
    if not key:
        raise SystemExit("Configura OPENAI_KEY en el entorno o en .env.")

    client = OpenAIChatCompletionClient(
        model="gpt-5.6-luna",
        api_key=key,
        reasoning_effort="none",
        max_tokens=1500,
        config=ModelConfig(supports_function_calling=True),
    )
    await run_agent_in_cli(client, provider="OpenAI")


if __name__ == "__main__":
    asyncio.run(main())
