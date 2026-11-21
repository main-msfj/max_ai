"""A session store of your own: four methods over any storage."""

import asyncio

from max_ai.agents import Agent
from max_ai.base.session_store import CoreSessionStore
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core.model.session import SessionInfo
from max_ai.types.run_context import RunContext


class InMemorySessionStore(CoreSessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.sessions: dict[tuple[str, str], tuple[RunContext, SessionInfo]] = {}

    async def _load(self, user_id: str, session_id: str) -> RunContext | None:
        kept = self.sessions.get((user_id, session_id))
        return kept[0] if kept else None

    async def _save(self, ctx: RunContext, info: SessionInfo) -> None:
        # ``info`` is the listing entry the base class already wrote: title, counts.
        self.sessions[(ctx.user_id, ctx.session_id or "")] = (ctx, info)

    async def _list(self, user_id: str) -> list[SessionInfo]:
        return [info for (owner, _), (_, info) in self.sessions.items() if owner == user_id]

    async def _delete(self, user_id: str, session_id: str) -> bool:
        return self.sessions.pop((user_id, session_id), None) is not None


async def main() -> None:
    store = InMemorySessionStore()
    ctx = await store.load("ana", "demo") or RunContext(user_id="ana", session_id="demo")
    agent = Agent(
        name="Assistant",
        description="An assistant with a session store of its own.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
    )
    try:
        state = await agent.run("Hello.", ctx)
        print(state.text)
        await store.save(ctx)
        print([info.title for info in await store.list_sessions("ana")])
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
