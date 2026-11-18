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
    assert box is not None and "bash(npm*)" in (await box.permissions())["allow"]
    await agent.close()
