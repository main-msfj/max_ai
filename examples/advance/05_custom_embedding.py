"""An embedding of your own: here, three counters of words by topic. No key needed."""

import asyncio

from max_ai.base.embedding import CoreEmbedding


class TopicEmbedding(CoreEmbedding):
    @property
    def model_id(self) -> str:
        return "topic-counter-v1"

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        topics = [
            {"bank", "interest", "money", "investment"},
            {"food", "recipe", "cooking", "ingredient"},
            {"code", "python", "programming", "software"},
        ]
        return [
            [float(sum(word in text.lower().split() for word in topic)) for topic in topics]
            for text in texts
        ]


async def main() -> None:
    vectors = await TopicEmbedding().embed(["investment and money", "cooking recipe"])
    print(vectors)


if __name__ == "__main__":
    asyncio.run(main())
