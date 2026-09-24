"""Run after setting MONGODB_URI: uv run --extra mongodb examples/mongodb_backends.py."""

import asyncio

from max_ai.capabilities.knowledge.mongodb import MongoDBKnowledgeRegistry
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry
from max_ai.core import KnowledgeBlock


async def main() -> None:
    async with MongoDBMemoryRegistry("demo_user", "demo_session") as memory:
        await memory.create_or_update("preferences", "Prefers answers in English.")
        print("Session memory:", await memory.get_context())

    async with MongoDBKnowledgeRegistry("demo_manual", "Search the project manual") as knowledge:
        await knowledge.upsert_block("intro", KnowledgeBlock(
            content="MaxAI stores memory by user and session in MongoDB.",
            metadata={"source": "manual", "section": "intro"},
        ))
        print("Knowledge:", await knowledge.search("MongoDB"))


if __name__ == "__main__":
    asyncio.run(main())
