"""One stateless Agent serves every user: memory is bound per run from the RunContext."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.memory.mongodb import MongoDBMemoryRegistry
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.core.model.llm import ModelConfig
from max_ai.errors.memory import MemoryError
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class ScriptedLLM:
    """Replays a queue of assistant messages; records every system prompt."""

    model = "fake"

    def __init__(self):
        self.queue: list[AssistantMessage] = []
        self.prompts: list[str] = []
        self.config = ModelConfig()
        self.generation_options = {"max_tokens": 200}

    def save(self, category: str, memory: str, call_id: str) -> None:
        self.queue += [
            AssistantMessage(source="llm", content="", tool_calls=[ToolCall(
                id=call_id, tool_name="create_or_update",
                parameters={"category": category, "memory": memory})]),
            AssistantMessage(source="llm", content="anotado"),
        ]

    async def run(self, *, ctx, prompts, tools=None, output_format=None, stream=False, **kwargs):
        self.prompts.append("\n".join(prompts.rendered_layers.values()))
        return ChatCompletionResult(
            message=self.queue.pop(0), usage=Usage(), model="fake", finish_reason="stop",
        )


def run_ctx(user: str, session: str) -> RunContext:
    return RunContext(user_id=user, session_id=session)


async def test_one_agent_keeps_each_users_memory_apart(tmp_path):
    llm = ScriptedLLM()
    memory = LocalMemoryRegistry(base_path=tmp_path / "data")  # backend only: no user
    async with Agent(name="demo", description="d", instructions="i", client=llm,
                     memory=memory, workspace=LocalWorkspace(root=tmp_path / "work")) as agent:
        llm.save("mascotas", "Tiene un perro llamado Toby.", "c1")
        await agent.run("mi perro se llama Toby", run_context=run_ctx("ana", "chat-1"))
        llm.save("dieta", "Es vegetariano.", "c2")
        await agent.run("soy vegetariano", run_context=run_ctx("beto", "chat-9"))

        llm.queue.append(AssistantMessage(source="llm", content="Toby"))
        await agent.run("¿cómo se llama mi perro?", run_context=run_ctx("ana", "chat-1"))

    files = tmp_path / "data" / "memory"
    assert json.loads((files / "ana" / "chat-1.json").read_text())["mascotas"]["memory"] == \
        "Tiene un perro llamado Toby."
    assert json.loads((files / "beto" / "chat-9.json").read_text())["dieta"]["memory"] == \
        "Es vegetariano."
    assert sorted(p.name for p in files.iterdir()) == ["ana", "beto"]

    # Each run's system prompt only ever carries that user's memories.
    ana_prompt = llm.prompts[-1]
    assert "Toby" in ana_prompt and "vegetariano" not in ana_prompt
    beto_prompts = llm.prompts[2:4]
    assert all("Toby" not in p for p in beto_prompts)


async def test_search_crosses_sessions_of_the_same_user_only(tmp_path):
    memory = LocalMemoryRegistry(base_path=tmp_path)
    await memory.bind("ana", "chat-1").create_or_update("mascotas", "perro Toby")
    await memory.bind("beto", "chat-9").create_or_update("mascotas", "perro Rex")
    found = await memory.bind("ana", "chat-2").search_memory("perro")
    assert [(r.session_id, r.memory) for r in found] == [("chat-1", "perro Toby")]


async def test_an_unbound_registry_refuses_to_touch_storage(tmp_path):
    memory = LocalMemoryRegistry(base_path=tmp_path)
    with pytest.raises(MemoryError, match="not bound"):
        await memory.get_context()
    with pytest.raises(MemoryError, match="not bound"):
        await memory.create_or_update("x", "y")


def test_bound_copies_are_independent_and_validated(tmp_path):
    memory = LocalMemoryRegistry(base_path=tmp_path)
    ana, beto = memory.bind("ana", "s1"), memory.bind("beto", "s2")
    assert (memory.user_id, ana.user_id, beto.user_id) == (None, "ana", "beto")
    with pytest.raises(ValueError, match="safe as filesystem paths"):
        memory.bind("../etc", "s1")


def test_bound_mongo_copies_share_one_client():
    memory = MongoDBMemoryRegistry(database="max_ai")
    memory._mongo_client, memory._collection = object(), object()  # as after connect()
    ana, beto = memory.bind("ana", "s1"), memory.bind("beto", "s2")
    assert ana._collection is beto._collection is memory._collection
    assert ana._scope() == {"user_id": "ana", "session_id": "s1"}
    assert beto._to_config().user_id == "beto" and memory._to_config().user_id is None


def test_the_serialized_agent_memory_has_no_user(tmp_path):
    config = LocalMemoryRegistry(base_path=tmp_path).serialize().config
    assert config["base_path"] == str(Path(tmp_path).resolve())
    assert "user_id" not in config and "session_id" not in config  # None values are dropped
