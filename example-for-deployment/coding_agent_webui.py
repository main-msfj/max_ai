"""The coding agent in the browser: the same agent as coding_agent.py, built
again from the YAML that script writes, and the same saved conversations.
Run coding_agent.py once first, then open http://127.0.0.1:8000."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from max_ai.agents.agent import Agent
from max_ai.capabilities.middleware import configure_langfuse
from max_ai.capabilities.session_store.local import LocalSessionStore
from max_ai.web import run_web

HERE = Path(__file__).parent


async def main() -> None:
    load_dotenv()
    tracing = configure_langfuse() if os.getenv("LANGFUSE_PUBLIC_KEY") else None
    agent = Agent.from_yaml(HERE / "agents" / "coding_agent.yaml")
    try:
        # The CLI's conversations show in the sidebar, and go on from here.
        store = LocalSessionStore(HERE / "coding_sessions")
        await run_web(agent, user_id="demo-coding", store=store)  # closes the agent at the end
    finally:
        if tracing:
            tracing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
