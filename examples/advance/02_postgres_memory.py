"""Minimal PostgreSQL memory backend. Install psycopg[binary]."""

import asyncio
import os

import psycopg

from max_ai.agents import Agent
from max_ai.base.memory import CoreMemoryRegistry, MemoryRecord, MemorySearchResult
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


class PostgresMemory(CoreMemoryRegistry):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.connection = None

    async def connect(self) -> None:
        self.connection = await psycopg.AsyncConnection.connect(self.url)
        await self.connection.execute(
            """CREATE TABLE IF NOT EXISTS example_memory (
                user_id TEXT, session_id TEXT, category TEXT, memory TEXT,
                updated_at TIMESTAMPTZ,
                PRIMARY KEY (user_id, session_id, category)
            )"""
        )

    async def disconnect(self) -> None:
        await self.connection.close()

    async def _read_session(self) -> list[MemoryRecord]:
        cursor = await self.connection.execute(
            "SELECT category, memory, updated_at FROM example_memory "
            "WHERE user_id = %s AND session_id = %s",
            (self.user_id, self.session_id),
        )
        return [MemoryRecord(category=row[0], memory=row[1], updated=row[2]) for row in await cursor.fetchall()]

    async def _write_memory(self, record: MemoryRecord) -> bool:
        cursor = await self.connection.execute(
            "SELECT 1 FROM example_memory WHERE user_id = %s AND session_id = %s AND category = %s",
            (self.user_id, self.session_id, record.category),
        )
        created = await cursor.fetchone() is None
        await self.connection.execute(
            """INSERT INTO example_memory VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, session_id, category) DO UPDATE SET
            memory = EXCLUDED.memory, updated_at = EXCLUDED.updated_at""",
            (self.user_id, self.session_id, record.category, record.memory, record.updated),
        )
        await self.connection.commit()
        return created

    async def _delete_memory(self, category: str) -> bool:
        cursor = await self.connection.execute(
            "DELETE FROM example_memory WHERE user_id = %s AND session_id = %s AND category = %s",
            (self.user_id, self.session_id, category),
        )
        await self.connection.commit()
        return cursor.rowcount > 0

    async def _search_memory(self, text: str) -> list[MemorySearchResult]:
        cursor = await self.connection.execute(
            """SELECT session_id, category, memory, updated_at FROM example_memory
            WHERE user_id = %s AND session_id <> %s AND memory ILIKE %s LIMIT 5""",
            (self.user_id, self.session_id, f"%{text}%"),
        )
        return [
            MemorySearchResult(session_id=row[0], category=row[1], memory=row[2], updated=row[3])
            for row in await cursor.fetchall()
        ]


async def main() -> None:
    memory = PostgresMemory(os.environ["DATABASE_URL"])
    await memory._ensure_connected()
    await memory.bind("user_ana", "first").create_or_update("preferences", "Prefers concise answers")
    agent = Agent(
        name="Assistant",
        description="An assistant with PostgreSQL memory.",
        instructions="Use memory to recall user preferences.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        memory=memory,
    )
    async with agent:
        response = await agent.run("What response style do I prefer?")
        print(response.final_text)
    await memory.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
