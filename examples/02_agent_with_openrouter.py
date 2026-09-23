"""Run the local demo Agent on OpenRouter free models, inside the Textual CLI.

From the repository root:
    .venv/bin/python -m examples.02_agent_with_openrouter

Loads OPENROUTER_API_KEY (or OPEN_ROUTER_KEY) from the environment/.env.
Free (``:free``) models get rate-limited upstream often, so a fallback list
lets OpenRouter switch models inside the same request. Tools, memory,
knowledge and skills come from ``examples/cli_agent.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from examples.cli_agent import run_agent_in_cli
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.core.model.llm import ModelConfig

# All support tool calling; check https://openrouter.ai/models?q=free for
# the current list — free models come and go.
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
FALLBACK_MODELS = ["qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"]


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Configura OPENROUTER_API_KEY en el entorno o en .env.")

    client = OpenRouterChatCompletionClient(
        model=MODEL,
        fallback_models=FALLBACK_MODELS,
        api_key=key,
        # Reasoning tokens count against max_tokens; keep them bounded.
        reasoning={"effort": "low"},
        max_tokens=4000,
        app_name="max_ai",
        config=ModelConfig(supports_function_calling=True),
    )
    await run_agent_in_cli(client, provider="OpenRouter")


if __name__ == "__main__":
    asyncio.run(main())
