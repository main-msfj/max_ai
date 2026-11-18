"""What the agent does around the loop: a turn that cannot start, the
sandbox made ready once per user, and a rule kept before any turn ran."""

from __future__ import annotations

import pytest

from max_ai.agents import Agent
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.skills.local import LocalSkills
from max_ai.capabilities.tools.bash import BashTool
from max_ai.core.event_type import FatalErrorEvent
from max_ai.core.messages import UserMessage
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord
from tests.middleware.fakes import FakeClient, say


async def test_a_turn_that_cannot_start_keeps_nothing_for_the_next_one(tmp_path):
    # The sandbox's folder is missing: getting ready fails before the loop.
    agent = Agent("a", "d", "i", FakeClient(say("listo")), sandbox=LocalSandbox(str(tmp_path / "work")))
    ctx = RunContext(user_id="u")
    events = []
    with pytest.raises(FileNotFoundError):
        async for event in agent.run_stream_events("hola", ctx):
            events.append(event)
    assert isinstance(events[-1], FatalErrorEvent) and ctx.inbox == []

    # The user tries again: what they said goes once, not twice.
    (tmp_path / "work").mkdir()
    state = await agent.run("hola", ctx)
    assert state.finish_reason == "stop"
    assert [m.text() for m in ctx.messages if isinstance(m, UserMessage)] == ["hola"]
    await agent.close()


class CountedSkills(LocalSkills):
    synced = 0

    async def sync(self, sandbox):
        CountedSkills.synced += 1
        return await super().sync(sandbox)


async def test_the_sandbox_is_made_ready_once_per_user(tmp_path):
    folder = tmp_path / "source" / "pdf"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: pdf\ndescription: Work with pdf files\n---\n")
    (tmp_path / "work").mkdir()
    CountedSkills.synced = 0
    agent = Agent(
        "a", "d", "i", FakeClient(), toolset=[BashTool()],
        sandbox=LocalSandbox(str(tmp_path / "work")),
        skills=CountedSkills(str(tmp_path / "source"), ["pdf"]),
    )
    ana = RunContext(user_id="ana")
    for task in ("uno", "dos", "tres"):
        await agent.run(task, ana)
    assert CountedSkills.synced == 1  # not once per message
    await agent.run("uno", RunContext(user_id="luis"))
    assert CountedSkills.synced == 2  # another user, another sandbox
    await agent.close()


async def test_always_keeps_the_rule_before_any_turn_has_run(tmp_path):
    (tmp_path / "work").mkdir()
    agent = Agent("a", "d", "i", FakeClient(), toolset=[BashTool()],
                  sandbox=LocalSandbox(str(tmp_path / "work")))
    # A session opened again: the call waits and no turn has run in this process.
    ctx = RunContext(user_id="u")
    ctx.tool_state.add(ToolCallRecord(
        id="a", tool_name="bash", parameters={"command": "npm test", "description": "d"}))
    assert await agent.approve(ctx, "a", always=True) is True
    box = await agent._sandbox(ctx)
    assert box is not None and (await box.permissions())["allow"] == ["bash(npm test)"]
    await agent.close()


# -------- deferred tools: few go loaded, many are searched ---------------------------
def records(query: str) -> str:
    """Search the customer records by name or email."""
    return query


async def test_few_deferred_tools_go_loaded_without_the_tools_that_find_them():
    from max_ai.capabilities.stacks.builder import PromptBuilder
    from max_ai.capabilities.tools.decorator import tool as as_tool

    few = as_tool(records, name="search_records", deferred=True)
    agent = Agent("a", "d", "i", FakeClient(say("listo")), toolset=[few])
    registry = await agent._registry()
    names = [tool.name for tool in registry.loaded_tools()]
    # The model gets it on every request, and nothing to search with.
    assert "search_records" in names and "search_tools" not in names and "call_tool" not in names
    assert registry.deferred_tools() == []
    assert PromptBuilder([], registry).tool_variables()["deferred_tools"] == []
    # The developer's tool is not changed: another agent may still defer it.
    assert few.deferred


async def test_many_deferred_tools_are_found_with_search_tools(monkeypatch):
    from max_ai.capabilities.tools.decorator import tool as as_tool
    from max_ai.config import setting

    monkeypatch.setattr(setting, "harness_deferred_tools_min_tokens", 10)
    many = as_tool(records, name="search_records", deferred=True)
    registry = await Agent("a", "d", "i", FakeClient(say("listo")), toolset=[many])._registry()
    names = [tool.name for tool in registry.loaded_tools()]
    assert "search_records" not in names and {"search_tools", "call_tool"} <= set(names)
    assert [tool.name for tool in registry.deferred_tools()] == ["search_records"]
