"""CLI as a session host: save every turn, --session, /resume, /new."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openrouter import OpenRouterChatCompletionClient
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.cli.app import MaxAIApp
from max_ai.cli.blocks import AssistantBlock, PastSummaryBlock, QuestionForm, UserBlock
from max_ai.core.messages import AssistantMessage, UserMessage
from max_ai.types.completions import LLMChunk, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.summary import Summary


class EchoLLM(OpenRouterChatCompletionClient):
    """Streams back ``re: <last user message>``."""

    def __init__(self) -> None:
        super().__init__(model="fake", api_key="x")

    async def complete_stream(self, messages, tools, output_format):  # type: ignore[override]
        last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        text = last if isinstance(last, str) else "".join(part.get("text", "") for part in last)
        reply = f"re: {text}"
        yield LLMChunk(content=reply, is_complete=False)
        yield LLMChunk(content="", is_complete=True, usage=Usage(tokens_input=10),
                                  message=AssistantMessage(source="llm", content=reply), finish_reason="stop")


@asynccontextmanager
async def make_agent(tmp_path: Path) -> AsyncIterator[Agent]:
    agent = Agent("demo", "d", "i", EchoLLM(), sandbox=LocalSandbox(str(tmp_path / "work")))
    try:
        yield agent
    finally:
        await agent.close()


async def idle(app, pilot, timeout: float = 10) -> None:
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if not app._busy:
            return
    raise AssertionError("still busy")


async def say(app, pilot, text: str) -> None:
    app.query_one("#prompt").text = text
    await pilot.press("enter")
    await idle(app, pilot)


def texts(app, kind) -> list[str]:
    if kind is UserBlock:
        return [str(b.render()).removeprefix("❯ ") for b in app.query(UserBlock)]
    return [b.text for b in app.query(kind)]


async def test_every_turn_is_saved_and_resumes_with_session(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="demo1")
        async with app.run_test() as pilot:
            await say(app, pilot, "hola")
            await say(app, pilot, "arma un scraper")
        saved = await store.load("marvin", "demo1")
        assert [m.text() for m in saved.messages] == ["hola", "re: hola", "arma un scraper",
                                                      "re: arma un scraper"]
        assert (await store.list_sessions("marvin"))[0].title == "hola"

        # A new process with --session demo1 redraws it and continues it.
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="demo1")
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            assert texts(app, UserBlock) == ["hola", "arma un scraper"]
            assert texts(app, AssistantBlock) == ["re: hola", "re: arma un scraper"]
            await say(app, pilot, "seguimos")
        assert len((await store.load("marvin", "demo1")).messages) == 6


async def test_unknown_session_starts_a_new_one_with_that_id(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="brand-new")
        async with app.run_test() as pilot:
            await say(app, pilot, "hola")
    assert (await store.load("marvin", "brand-new")) is not None


async def test_resume_picker_switches_conversation(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    for sid, first in (("old", "tema viejo"), ("recent", "tema reciente")):
        await store.save(RunContext(user_id="marvin", session_id=sid, messages=[
            UserMessage(source="marvin", content=first),
            AssistantMessage(source="agent", content=f"re: {first}"),
        ]))
        await asyncio.sleep(0.01)  # distinct updated_at

    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin")
        async with app.run_test() as pilot:
            app.query_one("#prompt").text = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.3)
            form = app.query_one(QuestionForm)
            assert form.questions[0].values == ["recent", "old"]  # newest first
            await pilot.press("down")  # highlight "old"
            await pilot.press("enter")
            await idle(app, pilot)
            await pilot.pause(0.2)
            assert app.context.session_id == "old"
            assert texts(app, UserBlock) == ["tema viejo"]


async def test_esc_cancels_the_resume_picker(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    await store.save(RunContext(user_id="marvin", session_id="other", messages=[
        UserMessage(source="marvin", content="otro tema")]))
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="mine")
        async with app.run_test() as pilot:
            app.query_one("#prompt").text = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.3)
            await pilot.press("escape")
            await idle(app, pilot)
            assert app.context.session_id == "mine" and not app._busy


async def test_new_starts_another_session_and_keeps_the_first(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="first")
        async with app.run_test() as pilot:
            await say(app, pilot, "primera")
            app.query_one("#prompt").text = "/new"
            await pilot.press("enter")
            await pilot.pause(0.2)
            second = app.context.session_id
            assert second != "first" and not app.query(UserBlock)
            await say(app, pilot, "segunda")
    titles = {s.session_id: s.title for s in await store.list_sessions("marvin")}
    assert titles == {"first": "primera", second: "segunda"}


async def test_a_compacted_session_shows_its_summary_on_resume(tmp_path):
    store = LocalSessionStore(tmp_path / "sessions")
    ctx = RunContext(user_id="marvin", session_id="long", messages=[
        UserMessage(source="marvin", content="último pedido")])
    ctx.summary = Summary(goal="armar un scraper")
    await store.save(ctx)
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, store=store, user_id="marvin", session_id="long")
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            block = app.query_one(PastSummaryBlock)
            assert "summarized" in block._header().plain and "armar un scraper" in block.summary


async def test_without_a_store_nothing_is_saved(tmp_path):
    async with make_agent(tmp_path) as agent:
        app = MaxAIApp(agent, user_id="marvin")
        async with app.run_test() as pilot:
            await say(app, pilot, "hola")
            app.query_one("#prompt").text = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.2)
            notes = " ".join(str(n.render()) for n in app.query("NoteLine"))
            assert "No session store configured" in notes
