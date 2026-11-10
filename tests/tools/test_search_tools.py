import logging

import pytest
from pydantic import BaseModel

from max_ai.base.embedding import CoreEmbedding
from max_ai.capabilities.tools.decorator import tool as as_tool
from max_ai.capabilities.tools.search_tool import SearchToolsTool
from max_ai.core.tool.registry import ToolRegistry
from max_ai.types.tool_call import ToolCallRecord

# Each dimension is one meaning, reached by words of either language.
MEANINGS = [
    {"web", "internet", "página"},
    {"weather", "forecast", "llover"},
    {"stay", "hotel", "dormir"},
]


class NoConfig(BaseModel):
    pass


class MeaningEmbedding(CoreEmbedding):
    component_schema = NoConfig

    @property
    def model_id(self) -> str:
        return "meanings"

    def _to_config(self) -> NoConfig:
        return NoConfig()

    @classmethod
    def _from_config(cls, config: NoConfig) -> "MeaningEmbedding":
        return cls()

    async def _embed(self, texts):
        return [[float(any(w in text.lower() for w in m)) for m in MEANINGS] for text in texts]


class BrokenEmbedding(MeaningEmbedding):
    async def _embed(self, texts):
        raise RuntimeError("no model")


async def exa_search(query: str) -> str:
    """Search the web and return pages with their text."""


async def get_weather(city: str) -> str:
    """Current forecast for a city."""


async def search_hotels(city: str) -> str:
    """Find places to stay in a city."""


async def send_email(to: str) -> str:
    """Send an email message to someone."""


def searcher(registry: ToolRegistry, embedding: CoreEmbedding | None) -> SearchToolsTool:
    tool = SearchToolsTool(registry)
    tool._embedding = embedding  # a fake model, or None for keywords only
    return tool


def catalog() -> ToolRegistry:
    deferred = [
        as_tool(fn, deferred=True) for fn in (exa_search, get_weather, search_hotels, send_email)
    ]
    return ToolRegistry([*deferred, as_tool(always_loaded)])


async def always_loaded(query: str) -> str:
    """Search everything, always sent to the model."""


async def call(tool: SearchToolsTool, **parameters):
    return await tool.execute(
        ToolCallRecord(id="s", tool_name="search_tools", parameters=parameters)
    )


def names(result) -> list[str]:
    return [found["name"] for found in result.result["tools"]]


@pytest.mark.asyncio
async def test_keywords_find_the_tool_and_return_its_schema():
    result = await call(searcher(catalog(), None), query="send an email")
    assert names(result) == ["send_email"]
    assert result.result["tools"][0]["parameters"]["required"] == ["to"]


@pytest.mark.asyncio
async def test_meaning_finds_what_keywords_miss():
    keywords = searcher(catalog(), None)
    hybrid = searcher(catalog(), MeaningEmbedding())
    assert names(await call(keywords, query="va a llover")) == []
    assert names(await call(hybrid, query="va a llover")) == ["get_weather"]
    assert names(await call(hybrid, query="dónde dormir")) == ["search_hotels"]


@pytest.mark.asyncio
async def test_both_rankings_agreeing_win():
    hybrid = searcher(catalog(), MeaningEmbedding())
    assert names(await call(hybrid, query="search the web"))[0] == "exa_search"


@pytest.mark.asyncio
async def test_names_load_several_tools_at_once():
    tool = searcher(catalog(), None)
    result = await call(tool, names=["get_weather", "exa_search"])
    assert names(result) == ["get_weather", "exa_search"] and "hint" not in result.result


@pytest.mark.asyncio
async def test_unknown_names_and_no_match_list_what_exists():
    tool = searcher(catalog(), None)
    missing = await call(tool, names=["exa_fetch"])
    assert names(missing) == [] and "No tool named exa_fetch" in missing.result["hint"]
    nothing = await call(tool, query="pizza")
    assert "exa_search, get_weather, search_hotels, send_email" in nothing.result["hint"]


@pytest.mark.asyncio
async def test_names_and_query_together_do_not_repeat():
    tool = searcher(catalog(), None)
    result = await call(tool, names=["send_email"], query="send an email")
    assert names(result) == ["send_email"]


@pytest.mark.asyncio
async def test_without_names_or_query_it_asks_for_one():
    result = await call(searcher(catalog(), None))
    assert not result.success


@pytest.mark.asyncio
async def test_never_more_than_the_limit():
    many = ToolRegistry()
    for number in range(8):

        async def records(x: str) -> str:
            """Search the stored records by text."""

        many.register(as_tool(records, name=f"search_{number}", deferred=True))
    result = await call(searcher(many, None), query="search")
    assert len(names(result)) == SearchToolsTool.LIMIT


@pytest.mark.asyncio
async def test_broken_embeddings_fall_back_to_keywords(caplog):
    tool = searcher(catalog(), BrokenEmbedding())
    with caplog.at_level(logging.WARNING):
        result = await call(tool, query="send an email")
    assert names(result) == ["send_email"]
    assert "embeddings unavailable" in caplog.text


@pytest.mark.asyncio
async def test_only_deferred_tools_are_searched_or_listed():
    tool = searcher(catalog(), None)
    assert "always_loaded" not in names(await call(tool, query="search everything"))
    assert "always_loaded" not in (await call(tool, query="pizza")).result["hint"]
