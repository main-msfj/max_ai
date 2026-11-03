"""The CLI drops a half-streamed answer when the model call starts over."""

from __future__ import annotations

import pytest

from max_ai.cli.app import MaxAIApp
from max_ai.cli.blocks import AssistantBlock, NoteLine
from max_ai.core.retry import RetryPolicy
from tests.cli.test_sessions import idle, say, texts
from tests.reasoning.test_stream_retry import FlakyStream, make_agent


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(RetryPolicy, "delay", lambda self, attempt, retry_after=None, **_: 0.0)


async def test_the_partial_answer_is_replaced_by_the_retry(tmp_path):
    async with make_agent(tmp_path, FlakyStream(sent=3)) as agent:
        app = MaxAIApp(agent, user_id="u", session_id="s")
        async with app.run_test() as pilot:
            await say(app, pilot, "¿resultado?")
            await idle(app, pilot)
            assert texts(app, AssistantBlock) == ["Hola, el resultado es 42."]
            notes = [str(n.render()) for n in app.query(NoteLine)]
            assert any("retrying in 0s (attempt 2 of 4)" in n for n in notes)
