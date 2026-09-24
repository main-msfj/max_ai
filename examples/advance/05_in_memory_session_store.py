"""Implement CoreSessionStore with an in-memory dictionary."""

import asyncio

from max_ai.agents import Agent
from max_ai.base.session_store import CoreSessionStore
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.model.session import SessionInfo
from max_ai.types.run_context import RunContext


class InMemorySessionStore(CoreSessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: dict[tuple[str, str], RunContext] = {}

    async def _load(self, user_id: str, session_id: str) -> RunContext | None:
        return self.sessions.get((user_id, session_id))

    async def _save(self, ctx: RunContext, info: SessionInfo) -> None:
        self.sessions[(ctx.user_id, ctx.session_id)] = ctx

    async def _list(self, user_id: str) -> list[SessionInfo]:
        return [
            SessionInfo(
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                title="Example session",
                message_count=len(ctx.messages),
            )
            for ctx in self.sessions.values()
            if ctx.user_id == user_id
        ]

    async def _delete(self, user_id: str, session_id: str) -> bool:
        return self.sessions.pop((user_id, session_id), None) is not None


async def main() -> None:
    store = InMemorySessionStore()
    context = await store.load("user_ana", "demo") or RunContext(user_id="user_ana", session_id="demo")
    agent = Agent(
        name="Assistant",
        description="An assistant with a custom session store.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
    )
    async with agent:
        response = await agent.run("Hello.", run_context=context)
        await store.save(response.context)
        print(response.final_text)


if __name__ == "__main__":
    asyncio.run(main())
