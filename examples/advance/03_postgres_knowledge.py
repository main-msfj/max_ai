"""Text knowledge source in PostgreSQL. Install psycopg[binary]."""

import asyncio
import os

import psycopg

from max_ai.agents import Agent
from max_ai.base.knowledge import CoreKnowledgeRegistry
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.core import KnowledgeBlock


class PostgresKnowledge(CoreKnowledgeRegistry):
    def __init__(self, url: str) -> None:
        super().__init__(name="office_notes", description="Notes stored in PostgreSQL.")
        self.url = url
        self.connection = None

    async def connect(self) -> None:
        self.connection = await psycopg.AsyncConnection.connect(self.url)
        await self.connection.execute(
            "CREATE TABLE IF NOT EXISTS example_knowledge (id TEXT PRIMARY KEY, content TEXT)"
        )
        await self.connection.execute(
            "INSERT INTO example_knowledge VALUES (%s, %s) ON CONFLICT DO NOTHING",
            ("note-1", "Office hours are Monday through Friday, 9 AM to 5 PM."),
        )
        await self.connection.commit()

    async def disconnect(self) -> None:
        await self.connection.close()

    async def search(self, query: str, limit: int = 5) -> list[KnowledgeBlock]:
        cursor = await self.connection.execute(
            "SELECT content FROM example_knowledge WHERE content ILIKE %s LIMIT %s",
            (f"%{query}%", limit),
        )
        return [KnowledgeBlock(content=row[0]) for row in await cursor.fetchall()]


async def main() -> None:
    knowledge = PostgresKnowledge(os.environ["DATABASE_URL"])
    agent = Agent(
        name="Assistant",
        description="Answer questions using PostgreSQL.",
        instructions="Search office_notes for questions about office hours.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
        knowledge=[knowledge],
    )
    async with agent:
        response = await agent.run("What are the office hours?")
        print(response.final_text)
    await knowledge.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
