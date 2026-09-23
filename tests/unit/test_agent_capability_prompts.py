"""Optional capabilities reach the model through real Agent runs."""

from types import SimpleNamespace

import pytest

from max_ai.agents.agent import Agent
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.stacks import (
    KnowledgeLayer,
    MemoryLayer,
    SessionStateLayer,
    SkillsLayer,
)
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import AssistantMessage, ToolCall
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class RecordingClient:
    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self, calls=()):
        self.calls = list(calls)
        self.prompts = []
        self.tools = []

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.prompts.append(prompts)
        self.tools = tools
        calls, self.calls = self.calls, []
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content="" if calls else "done",
                                     tool_calls=calls),
            usage=Usage(), model=self.model,
            finish_reason="tool_calls" if calls else "stop",
        )


def make_agent(tmp_path, client, **kwargs):
    return Agent(name="test", description="test", instructions="Help.",
                 client=client, workspace=LocalWorkspace(root=tmp_path / "work"),
                 **kwargs)


@pytest.mark.asyncio
async def test_no_capabilities(tmp_path):
    client = RecordingClient()
    async with make_agent(tmp_path, client) as agent:
        await agent.run("hello")
        assert agent.skills is None
        names = {type(layer).__name__ for layer in client.prompts[0].stack}
        assert names == {"AgentPolicyLayer", "TaskAnalysisLayer", "RenderingLayer", "SessionStateLayer"}
        assert client.prompts[0].rendered_layers[SessionStateLayer] == ""  # empty until needed
        assert "get_context" not in {tool.name for tool in agent._registry.all_tools()}


@pytest.mark.asyncio
async def test_memory_tools_execute_and_snapshot_refreshes(tmp_path):
    memory = LocalMemoryRegistry("u", "s", tmp_path)
    client = RecordingClient([ToolCall(id="save", tool_name="create_or_update",
        parameters={"category": "project", "memory": "First line\nSecond line"})])
    ctx = RunContext(user_id="u", session_id="s")
    async with make_agent(tmp_path, client, memory=memory) as agent:
        await agent.run("remember", run_context=ctx)
        assert (await memory.get_context())[0].memory == "First line\nSecond line"
        await agent.run("recall", run_context=ctx)
        assert "First line\nSecond line" in client.prompts[-1].rendered_layers[MemoryLayer]
        assert agent._registry.runs_on_host("create_or_update")
        # Another session gets its own, empty memory: never "s"'s memories.
        await agent.run("other session", run_context=RunContext(user_id="u", session_id="other"))
        assert "First line" not in client.prompts[-1].rendered_layers[MemoryLayer]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,count", [(MemoryToolMode.NONE, 0),
    (MemoryToolMode.READ_ONLY, 3), (MemoryToolMode.FULL, 5)])
async def test_memory_modes(tmp_path, mode, count):
    memory = LocalMemoryRegistry("u", "s", tmp_path, tool_mode=mode)
    await memory.create_or_update("preference", "English")
    client = RecordingClient()
    async with make_agent(tmp_path, client, memory=memory) as agent:
        await agent.run("hello", run_context=RunContext(user_id="u", session_id="s"))
        prompt = client.prompts[0]
        assert len(prompt.variables["memory_tools"]) == count
        assert "English" in prompt.rendered_layers[MemoryLayer]
        assert ("Call create_or_update" in prompt.rendered_layers[MemoryLayer]) == (count == 5)


@pytest.mark.asyncio
async def test_skills_and_knowledge(tmp_path, monkeypatch):
    source = tmp_path / "source"
    skill_dir = source / "writing"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: writing\ndescription: Write reports\n---\nFull instructions.")
    monkeypatch.setattr(LocalSkillRegistry, "_resolve_cache_root", staticmethod(lambda: tmp_path / "cache"))
    skills = LocalSkillRegistry(source=source, skills=["writing"])
    knowledge = LocalKnowledgeRegistry("docs", "Search documentation", tmp_path)
    client = RecordingClient([ToolCall(id="search", tool_name="search_docs",
                                     parameters={"query": "example"})])
    async with make_agent(tmp_path, client, skills=skills, knowledge=[knowledge]) as agent:
        ctx = RunContext(user_id="u", session_id="s")
        await agent.run("help", run_context=ctx)
        prompt = client.prompts[0]
        assert "Write reports" in prompt.rendered_layers[SkillsLayer]
        assert "Full instructions." not in prompt.rendered_layers[SkillsLayer]
        assert "search_docs" in prompt.rendered_layers[KnowledgeLayer]
        assert agent._registry.runs_on_host("search_docs")
        directory = agent.workspace.materialize("u", "s")
        assert (directory.skill_dir / "writing" / "SKILL.md").is_file()
        assert len(client.prompts) == 2


def test_duplicate_capability_tool_fails(tmp_path):
    def get_context():
        return "unrelated"
    with pytest.raises(ValueError, match="duplicate tool name"):
        make_agent(tmp_path, RecordingClient(), toolset=[get_context],
                   memory=LocalMemoryRegistry("u", "s", tmp_path))
