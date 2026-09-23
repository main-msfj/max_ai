"""LocalSessionStore: hosts save and resume whole conversations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.session_store import LocalSessionStore
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.capabilities.tools.plan import AgentPlan
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import HARNESS_SOURCE, AssistantMessage, ToolCall, UserMessage
from max_ai.core.model.llm import ModelConfig
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


def conversation(user: str = "alice", session: str = "s1", first: str = "arma un scraper") -> RunContext:
    ctx = RunContext(user_id=user, session_id=session, messages=[
        UserMessage(source=HARNESS_SOURCE, content="max iterations reached"),
        UserMessage(source=user, content=first),
        AssistantMessage(source="agent", content="listo"),
    ])
    ctx.compaction.state["summary"] = {"summary": "resumen previo"}
    ctx.compaction.compactions = 2
    ctx.plan = AgentPlan(rationale="r", steps=[{"id": 1, "description": "paso", "status": "active"}])
    return ctx


async def test_round_trip_keeps_the_whole_conversation(tmp_path):
    store = LocalSessionStore(tmp_path)
    ctx = conversation()
    info = await store.save(ctx)
    assert (info.title, info.message_count, info.compactions) == ("arma un scraper", 3, 2)

    loaded = await LocalSessionStore(tmp_path).load("alice", "s1")  # e.g. another process
    assert loaded.model_dump() == ctx.model_dump()
    assert loaded.compaction.state["summary"] == {"summary": "resumen previo"}
    assert loaded.plan.steps[0].status == "active"


async def test_missing_session_is_none(tmp_path):
    assert await LocalSessionStore(tmp_path).load("alice", "nope") is None


async def test_users_never_see_each_other(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.save(conversation("alice", "s1"))
    await store.save(conversation("bob", "s1", first="otra cosa"))
    assert [s.title for s in await store.list_sessions("alice")] == ["arma un scraper"]
    assert (await store.load("bob", "s1")).messages[1].text() == "otra cosa"

    # A file copied into someone else's folder is refused, not served.
    alice_file = tmp_path / "alice" / "s1.json"
    (tmp_path / "bob" / "s2.json").write_text(alice_file.read_text())
    with pytest.raises(ValueError, match="does not belong"):
        await store.load("bob", "s2")


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 129, "sp ace"])
async def test_unsafe_ids_are_rejected(tmp_path, bad):
    store = LocalSessionStore(tmp_path)
    with pytest.raises(ValueError, match="Invalid"):
        await store.load(bad, "s1")
    with pytest.raises(ValueError, match="Invalid"):
        await store.save(RunContext(user_id="alice", session_id=bad))


async def test_list_is_newest_first_with_a_limit(tmp_path):
    store = LocalSessionStore(tmp_path)
    for n in range(3):
        await store.save(conversation(session=f"s{n}", first=f"tarea {n}"))
    listed = await store.list_sessions("alice")
    assert [s.session_id for s in listed] == ["s2", "s1", "s0"]
    assert len(await store.list_sessions("alice", limit=2)) == 2
    assert await store.list_sessions("nobody") == []


async def test_long_titles_are_shortened(tmp_path):
    info = await LocalSessionStore(tmp_path).save(conversation(first="palabra " * 50))
    assert len(info.title) == 80 and info.title.endswith("…")


async def test_delete_and_no_temp_files_left(tmp_path):
    store = LocalSessionStore(tmp_path)
    await store.save(conversation())
    assert not list(tmp_path.rglob("*.tmp"))
    assert await store.delete("alice", "s1") is True
    assert await store.delete("alice", "s1") is False
    assert await store.load("alice", "s1") is None and await store.list_sessions("alice") == []


def test_store_is_a_serializable_component(tmp_path):
    store = LocalSessionStore(tmp_path)
    component = store.serialize()
    assert component.provider == "maxai.session_store.LocalSessionStore"
    assert LocalSessionStore.deserialize(component).base_path == Path(tmp_path)


# -------- the production flow: load → run → save, across "processes" ------------------
def send_email(to: str) -> dict:
    """Send an email."""
    return {"sent": to}


class ScriptedLLM:
    model = "fake"

    def __init__(self, *messages):
        self.messages = list(messages)
        self.config = ModelConfig()
        self.generation_options = {"max_tokens": 200}

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        message = self.messages.pop(0)
        return ChatCompletionResult(message=message, usage=Usage(), model="fake", finish_reason="stop")


def make_agent(root: Path, *messages) -> Agent:
    return Agent(
        name="demo", description="d", instructions="i", client=ScriptedLLM(*messages),
        toolset=[FunctionAsTool(send_email, approval_mode=ToolApprovalMode.ASK_APPROVED)],
        workspace=LocalWorkspace(root=root / "work"),
    )


async def test_a_paused_turn_resumes_in_another_process(tmp_path):
    call = AssistantMessage(source="llm", content="", tool_calls=[
        ToolCall(id="e1", tool_name="send_email", parameters={"to": "a@b.c"})])
    done = AssistantMessage(source="llm", content="Email enviado.")

    # Process 1: the turn pauses waiting for approval; the host saves it.
    async with make_agent(tmp_path, call) as agent:
        response = await agent.run("manda el email", run_context=RunContext(user_id="alice", session_id="s1"))
        assert response.needs_approval
        await LocalSessionStore(tmp_path / "sessions").save(response.context)

    # Process 2: a fresh store and agent load it, approve and finish.
    ctx = await LocalSessionStore(tmp_path / "sessions").load("alice", "s1")
    pending = next(r for r in ctx.tool_state.records.values() if not r.is_consumed)
    ctx.tool_state.apply_approval(pending.id, True)
    async with make_agent(tmp_path, done) as agent:
        response = await agent.resume(ctx)
    assert response.finish_reason == "stop"
    assert response.context.messages[-1].text() == "Email enviado."
    saved = json.loads((tmp_path / "sessions" / "alice" / "s1.meta.json").read_text())
    assert saved["title"] == "manda el email"
