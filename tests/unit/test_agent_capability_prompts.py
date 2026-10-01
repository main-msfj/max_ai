"""Optional capabilities reach the model through real Agent runs."""

from types import SimpleNamespace

import pytest

from max_ai.agents.agent import COMMANDS_OFF, Agent
from max_ai.base.memory import MemoryToolMode
from max_ai.capabilities.executor.docker import DockerExecutor
from max_ai.capabilities.executor.local import LocalExecutor
from max_ai.capabilities.executor.modal import ModalExecutor
from max_ai.capabilities.knowledge.local import LocalKnowledgeRegistry
from max_ai.capabilities.memory.local import LocalMemoryRegistry
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.capabilities.stacks import (
    AgentPolicyLayer,
    KnowledgeLayer,
    MemoryLayer,
    SessionStateLayer,
    SkillsLayer,
    TaskAnalysisLayer,
)
from max_ai.capabilities.tools.bash import BashTool
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


async def policy_prompt(agent) -> str:
    prompts = await agent._prompts(RunContext(user_id="u", session_id="s"))
    return prompts.rendered_layers[AgentPolicyLayer]


@pytest.mark.asyncio
async def test_the_prompt_says_where_commands_run(tmp_path):
    modal = make_agent(tmp_path, RecordingClient(), executor=ModalExecutor())
    text = await policy_prompt(modal)
    assert "Execution environment:" in text
    assert "Install what a script needs with pip, uv or npm before running it; other sites are blocked" in text

    docker = make_agent(tmp_path, RecordingClient(), executor=DockerExecutor(network="internet"))
    assert "isolated Docker container as a non-root user" in await policy_prompt(docker)
    assert "It has internet access." in await policy_prompt(docker)

    from datetime import UTC, datetime
    assert f"Today is {datetime.now(UTC).date().isoformat()} (UTC)." in text

    # The local executor runs on the host: nothing to add.
    assert "Execution environment" not in await policy_prompt(make_agent(tmp_path, RecordingClient()))


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
        # The user's other sessions see it too; other users never do.
        await agent.run("other session", run_context=RunContext(user_id="u", session_id="other"))
        assert "First line" in client.prompts[-1].rendered_layers[MemoryLayer]
        await agent.run("other user", run_context=RunContext(user_id="someone", session_id="s"))
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
        directory = agent.workspace.materialize("u", "s")
        assert (directory.skill_dir / "writing" / "SKILL.md").is_file()
        assert len(client.prompts) == 2


def test_duplicate_capability_tool_fails(tmp_path):
    def get_context():
        return "unrelated"
    with pytest.raises(ValueError, match="duplicate tool name"):
        make_agent(tmp_path, RecordingClient(), toolset=[get_context],
                   memory=LocalMemoryRegistry("u", "s", tmp_path))


# -------- Commands on the host --------------------------------------------------
def has_bash(agent) -> bool:
    return any(tool.name == "bash" for tool in agent.tools)


def test_bash_only_when_the_executor_runs_commands(tmp_path):
    assert not has_bash(make_agent(tmp_path, RecordingClient()))  # LocalExecutor: off
    assert has_bash(make_agent(tmp_path, RecordingClient(), executor=LocalExecutor(allow_commands=True)))
    assert has_bash(make_agent(tmp_path, RecordingClient(), executor=DockerExecutor()))
    with pytest.raises(ValueError, match="allow_commands=True"):
        make_agent(tmp_path, RecordingClient(), toolset=[BashTool()])


@pytest.mark.asyncio
async def test_without_commands_the_prompt_never_mentions_them(tmp_path, monkeypatch, caplog):
    skill = tmp_path / "source" / "writing"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: writing\ndescription: Write reports\n---\nSteps.")
    monkeypatch.setattr(LocalSkillRegistry, "_resolve_cache_root", staticmethod(lambda: tmp_path / "cache"))
    skills = LocalSkillRegistry(source=tmp_path / "source", skills=["writing"])
    agent = make_agent(tmp_path, RecordingClient(), skills=skills)
    assert COMMANDS_OFF in caplog.text
    async with agent:
        await agent.prepare()  # loads the skills
        prompts = await agent._prompts(RunContext(user_id="u", session_id="s"))
        text = "".join(prompts.rendered_layers[layer] for layer in (SkillsLayer, AgentPolicyLayer, TaskAnalysisLayer))
        assert "skills/writing/SKILL.md" in text
        assert "bash" not in text.lower() and "/tmp" not in text  # only what it has


@pytest.mark.asyncio
async def test_on_the_host_even_allowed_commands_ask(tmp_path):
    client = RecordingClient([ToolCall(id="c1", tool_name="bash",
                                       parameters={"command": "pwd", "description": "where am I"})])
    agent = make_agent(tmp_path, client, executor=LocalExecutor(allow_commands=True))
    assert "Bash(pwd)" in agent.policy.allow  # allowed by the policy...
    async with agent:
        response = await agent.run("where?", run_context=RunContext(user_id="u", session_id="s"))
    assert response.finish_reason == "approval_needed"  # ...but the host asks anyway


@pytest.mark.asyncio
async def test_a_denied_command_never_runs_and_says_why(tmp_path):
    marker = tmp_path / "ran"
    client = RecordingClient([ToolCall(id="c1", tool_name="bash", parameters={
        "command": f"pwd && sudo touch {marker}", "description": "try sudo"})])
    agent = make_agent(tmp_path, client, executor=LocalExecutor(allow_commands=True))
    ctx = RunContext(user_id="u", session_id="s")
    async with agent:
        await agent.run("go", run_context=ctx)
    assert not marker.exists()
    result = next(m for m in ctx.messages if getattr(m, "tool_name", None) == "bash")
    assert "Denied by policy: Bash(sudo:*)" in str(result.content)
