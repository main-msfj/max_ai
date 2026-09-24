"""Create a teaching embedding with three word counters."""

import asyncio

from max_ai.base.embedding import CoreEmbedding


class TopicEmbedding(CoreEmbedding):
    @property
    def model_id(self) -> str:
        return "topic-counter-v1"

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        temas = [
            {"bank", "interest", "money", "investment"},
            {"food", "recipe", "cooking", "ingredient"},
            {"code", "python", "programming", "software"},
        ]
        return [
            [float(sum(palabra in texto.lower().split() for palabra in tema)) for tema in temas]
            for texto in texts
        ]


async def main() -> None:
    vectors = await TopicEmbedding().embed(["investment and money", "cooking recipe"])
    print(vectors)


if __name__ == "__main__":
    asyncio.run(main())
