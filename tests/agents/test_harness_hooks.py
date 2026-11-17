from max_ai.base.loop import LoopState
from max_ai.base.middleware import CoreMiddleware
from max_ai.base.tools import ToolContext
from max_ai.capabilities.memory import Memory
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.filesystem import WriteFileTool
from max_ai.core.harness.hooks import HarnessLoopHook, HarnessToolHook
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult


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


async def test_middlewares_see_each_tool_and_one_that_fails_stops_nothing(tmp_path):
    seen = []

    class Watch(CoreMiddleware):
        async def on_tool_request(self, mw, request):
            seen.append(("before", request.tool_name))

        async def on_tool_response(self, mw, request, result):
            seen.append(("after", result.success))
            return result

    class Broken(CoreMiddleware):
        async def on_tool_request(self, mw, request):
            raise RuntimeError("boom")

        async def on_tool_response(self, mw, request, result):
            raise RuntimeError("boom")

    ctx, _ = tool_ctx(tmp_path)
    hook, tool = HarnessToolHook(middlewares=[Broken(), Watch()]), WriteFileTool()
    assert await hook.before(tool, record("notes.md"), ctx) is None
    result = await hook.after(tool, record("notes.md"), ctx, ToolResult.success_result("c1", "ok"))
    assert result.success and seen == [("before", "write_file"), ("after", True)]


async def test_the_loop_hook_renews_the_memory_index_after_a_compaction(tmp_path):
    class Fake(Memory):
        async def index(self, sandbox):
            return "- fact.md: likes tea"

    ctx, state = RunContext(user_id="u", session_id="s"), LoopState()
    hook = HarnessLoopHook(memory=Fake(), sandbox=LocalSandbox(str(tmp_path)))
    await hook.after_compact(ctx, state)
    assert ctx.memory_index == "- fact.md: likes tea"
    assert await HarnessLoopHook().notes(ctx, state) == []
