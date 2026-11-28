"""bash in the background, and the two tools that look at a task and stop it."""

import asyncio
from pathlib import Path

import pytest

from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.bash import BashTool
from max_ai.capabilities.tools.tasks import StopTaskTool, TaskOutputTool
from max_ai.core.tool import ToolDispatcher, ToolHook, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


@pytest.fixture
async def sandbox(tmp_path):
    bound = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await bound.prepare()
    return bound


@pytest.fixture
def ctx():
    return RunContext(user_id="user-1", session_id="s1")


class Approves(ToolHook):
    """Stands for the harness saying yes to every command."""

    async def permission(self, tool, record, tool_ctx):
        return True


async def call(sandbox, ctx, name, hook=None, **parameters):
    if name == "bash":
        parameters["description"] = "test"
    tools = ToolRegistry([BashTool(), TaskOutputTool(), StopTaskTool()])
    record = ToolCallRecord(tool_name=name, parameters=parameters)
    return await ToolDispatcher(tools, sandbox, hook=hook or Approves()).dispatch(record, ctx)


async def until_ended(sandbox, task_id):
    for _ in range(100):
        (task,) = await sandbox.tasks("s1", task_id)
        if task.state != "running":
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the task never ended")


async def test_a_background_command_returns_at_once_with_its_task(sandbox, ctx):
    await call(sandbox, ctx, "bash", command="mkdir -p repo && cd repo")
    started = await call(sandbox, ctx, "bash", command="pwd -P; sleep 30", background=True)
    assert started.success and started.result.startswith("Started in the background as task 1.")
    assert ctx.tasks == {"1": "pwd -P; sleep 30"}
    # It started where the last command ended.
    await asyncio.sleep(0.3)
    seen = await call(sandbox, ctx, "task_output", task_id="1")
    assert seen.result.startswith("task 1: running\n") and seen.result.endswith("/repo")
    await call(sandbox, ctx, "stop_task", task_id="1")


async def test_a_command_that_waits_leaves_no_task(sandbox, ctx):
    await call(sandbox, ctx, "bash", command="echo hi")
    await call(sandbox, ctx, "bash", command="echo hi", background=False)
    assert ctx.tasks == {} and not (Path(str(sandbox.layout.tasks)) / "s1").exists()
    assert (await call(sandbox, ctx, "task_output")).result == TaskOutputTool.NONE


async def test_a_background_command_asks_like_any_command(sandbox, ctx):
    waiting = await call(sandbox, ctx, "bash", hook=ToolHook(), command="sleep 30", background=True)
    assert waiting is None and ctx.tasks == {}
    assert await sandbox.tasks("s1") == []


async def test_one_task_too_many_is_told_to_the_model(sandbox, ctx):
    for _ in range(3):
        await call(sandbox, ctx, "bash", command="sleep 30", background=True)
    refused = await call(sandbox, ctx, "bash", command="sleep 30", background=True)
    assert not refused.success and "already has 3 background tasks running" in refused.error
    assert sorted(ctx.tasks) == ["1", "2", "3"]
    for task_id in list(ctx.tasks):
        await call(sandbox, ctx, "stop_task", task_id=task_id)


async def test_the_tasks_are_listed_and_one_is_shown_with_its_output(sandbox, ctx):
    await call(sandbox, ctx, "bash", command="sleep 30", background=True)
    await call(sandbox, ctx, "bash", command="echo one; echo two; exit 3", background=True)
    await until_ended(sandbox, "2")
    listed = await call(sandbox, ctx, "task_output")
    assert listed.result == (
        "task 1: running | sleep 30\ntask 2: done, exit code 3 | echo one; echo two; exit 3"
    )
    # The model saw that task 2 ended: only the one that runs is still expected.
    assert list(ctx.tasks) == ["1"]
    shown = await call(sandbox, ctx, "task_output", task_id="2")
    assert shown.result == "task 2: done, exit code 3\none\ntwo"
    await call(sandbox, ctx, "stop_task", task_id="1")


async def test_a_task_is_stopped_and_forgotten(sandbox, ctx):
    await call(sandbox, ctx, "bash", command="echo begin; sleep 30", background=True)
    await asyncio.sleep(0.3)
    stopped = await call(sandbox, ctx, "stop_task", task_id="1")
    assert stopped.result == "task 1: stopped" and ctx.tasks == {}
    # What it printed is still there, and stopping it again changes nothing.
    assert (await call(sandbox, ctx, "task_output", task_id="1")).result == "task 1: stopped\nbegin"
    again = await call(sandbox, ctx, "stop_task", task_id="1")
    assert again.result == "It had already ended, nothing was stopped. task 1: stopped"


async def test_a_task_that_is_not_there_is_an_error_the_model_reads(sandbox, ctx):
    for name in ("task_output", "stop_task"):
        missing = await call(sandbox, ctx, name, task_id="9")
        assert not missing.success and "there is no task 9" in missing.error


async def test_the_two_tools_never_ask_the_user(sandbox, ctx):
    # No hook says yes here: bash would wait, these two run.
    listed = await call(sandbox, ctx, "task_output", hook=ToolHook())
    assert listed is not None and listed.result == TaskOutputTool.NONE
    missing = await call(sandbox, ctx, "stop_task", hook=ToolHook(), task_id="1")
    assert missing is not None and not missing.success
