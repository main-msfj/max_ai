import asyncio

from max_ai.base.loop import LoopState
from max_ai.base.tools import ToolContext
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import WriteFileTool
from max_ai.core.harness.hooks import HarnessLoopHook, HarnessToolHook
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord


def record(path, name="write_file"):
    return ToolCallRecord(id="c1", tool_name=name, parameters={"path": path, "content": "x"})


def tool_ctx(tmp_path):
    sandbox = LocalSandbox(str(tmp_path))
    return ToolContext(RunContext(user_id="u", session_id="s"), sandbox), sandbox


async def test_a_read_only_memory_is_not_written_by_the_file_tools(tmp_path):
    ctx, sandbox = tool_ctx(tmp_path)
    hook, tool = HarnessToolHook(memory=Memory(read_only=True)), WriteFileTool()
    inside = str(sandbox.layout.memory / "fact.md")
    denied = await hook.before(tool, record(inside), ctx)
    assert denied is not None and not denied.success and "read-only" in denied.error
    # A relative path that climbs into the memory is caught too; the workspace is free.
    assert await hook.before(tool, record("../memory/fact.md"), ctx) is not None
    assert await hook.before(tool, record("notes.md"), ctx) is None
    assert await HarnessToolHook(memory=Memory()).before(tool, record(inside), ctx) is None


async def test_the_loop_hook_renews_the_memory_index_after_a_compaction(tmp_path):
    class Fake(Memory):
        async def index(self, sandbox):
            return "- fact.md: likes tea"

    ctx, state = RunContext(user_id="u", session_id="s"), LoopState()
    hook = HarnessLoopHook(memory=Fake(), sandbox=LocalSandbox(str(tmp_path)))
    await hook.after_compact(ctx, state)
    assert ctx.memory_index == "- fact.md: likes tea"
    assert await HarnessLoopHook().notes(ctx, state) == []


async def test_a_background_task_that_ended_is_told_once(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("u")
    await sandbox.prepare()
    hook, state = HarnessLoopHook(sandbox=sandbox), LoopState()
    ctx = RunContext(user_id="u", session_id="s")
    ctx.tasks[await sandbox.start_task("s", "sleep 30", str(tmp_path))] = "sleep 30"
    ctx.tasks[await sandbox.start_task("s", "echo one; echo two; exit 3", str(tmp_path))] = "echo"
    for _ in range(100):
        if (await sandbox.tasks("s", "2"))[0].state != "running":
            break
        await asyncio.sleep(0.05)
    # The one that ended is told with how it ended and what it printed; the other stays.
    assert await hook.notes(ctx, state) == [
        "Background task 2: done, exit code 3.\nCommand: echo\nIts last lines:\none\ntwo"
    ]
    assert ctx.tasks == {"1": "sleep 30"}
    assert await hook.notes(ctx, state) == []
    # Stopped from outside, and one whose folder is gone: each told, then forgotten.
    await sandbox.stop_task("s", "1")
    ctx.tasks["7"] = "make all"
    assert await hook.notes(ctx, state) == [
        "Background task 1: stopped.\nCommand: sleep 30\nIts last lines:\n(it printed nothing)",
        "Background task 7 is no longer in the sandbox; how it ended was lost.\nCommand: make all",
    ]
    assert ctx.tasks == {}


async def test_the_sandbox_is_asked_about_tasks_only_when_some_are_pending(tmp_path, caplog):
    class Counts(LocalSandbox):
        asked = 0

        async def tasks(self, session_id, task_id=None, lines=20):
            type(self).asked += 1
            raise RuntimeError("the sandbox is down")

    hook, state = HarnessLoopHook(sandbox=Counts(str(tmp_path))), LoopState()
    ctx = RunContext(user_id="u", session_id="s")
    assert await hook.notes(ctx, state) == [] and Counts.asked == 0
    # A sandbox that fails does not break the turn, and the task is asked for again.
    ctx.tasks["1"] = "sleep 30"
    assert await hook.notes(ctx, state) == [] and await hook.notes(ctx, state) == []
    assert Counts.asked == 2 and ctx.tasks == {"1": "sleep 30"}
