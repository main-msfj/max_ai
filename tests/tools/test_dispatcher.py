"""ToolDispatcher used on its own: a registry, a RunContext and, when a tool
needs it, a sandbox. No Agent."""

import asyncio

from max_ai.base.tools import ToolContext
from max_ai.capabilities.sandbox.local import LocalSandbox
from max_ai.capabilities.tools.function_as_tool import FunctionAsTool
from max_ai.core.event_type import ToolApprovalEvent, ToolCallResponseEvent
from max_ai.core.messages import ToolMessage
from max_ai.core.tool import ToolDispatcher, ToolHook, ToolRegistry
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode

AUTO = ToolApprovalMode.AUTO_APPROVED


def add(a: int, b: int) -> int:
    """Add two whole numbers and return the sum of them."""
    return a + b


def delete_everything() -> str:
    """Delete every file of the project, for ever and with no way back."""
    return "gone"


class WhereAmI(FunctionAsTool):
    """A tool that runs in the sandbox and says what it was handed."""

    sandbox = True

    async def execute(self, tool_request, tool_context=None, cancellation_token=None):
        found = await tool_context.sandbox.execute_command("echo $USER_COLOR")
        return ToolResult.success_result(
            tool_request.id,
            {
                "user": tool_context.user_id,
                "session": tool_context.session_id,
                "color": found.stdout.strip(),
                "db": tool_context.deps.get("db"),
            },
        )


def where_am_i() -> str:
    """Say which user and session this call belongs to, from the sandbox."""
    return ""


def call(name: str, **parameters) -> ToolCallRecord:
    return ToolCallRecord(tool_name=name, parameters=parameters)


def registry() -> ToolRegistry:
    return ToolRegistry(
        [
            FunctionAsTool(add, approval_mode=AUTO),
            FunctionAsTool(delete_everything),
            WhereAmI(where_am_i, approval_mode=AUTO),
        ]
    )


# -------- on its own -------------------------------------------------------------
async def test_a_host_tool_runs_without_a_sandbox():
    record = call("add", a=2, b=3)
    result = await ToolDispatcher(registry()).dispatch(record, RunContext())
    assert result.success and result.result == 5
    assert record.is_consumed


async def test_an_unknown_tool_and_bad_parameters_are_answered_not_raised():
    dispatcher = ToolDispatcher(registry())
    unknown = await dispatcher.dispatch(call("nope"), RunContext())
    wrong = await dispatcher.dispatch(call("add", a="two", b=3), RunContext())
    assert not unknown.success and "Unknown tool" in unknown.error
    assert not wrong.success


async def test_a_sandbox_tool_without_a_sandbox_says_so_and_does_not_run():
    result = await ToolDispatcher(registry()).dispatch(call("where_am_i"), RunContext())
    assert not result.success and "needs a sandbox" in result.error


async def test_a_sandbox_tool_gets_the_users_sandbox_and_the_conversation(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1", {"USER_COLOR": "blue"})
    dispatcher = ToolDispatcher(registry(), sandbox, deps={"db": "my-db"})
    ctx = RunContext(user_id="user-1", session_id="s1")
    result = await dispatcher.dispatch(call("where_am_i"), ctx)
    assert result.result == {"user": "user-1", "session": "s1", "color": "blue", "db": "my-db"}


# -------- approval ---------------------------------------------------------------
async def test_a_call_that_needs_approval_waits_and_runs_once_approved():
    dispatcher, ctx, events = ToolDispatcher(registry()), RunContext(), []
    record = call("delete_everything")

    assert await dispatcher.dispatch(record, ctx, events.append) is None
    assert record.is_pending_approval
    assert any(isinstance(event, ToolApprovalEvent) for event in events)

    record.approve("yes")
    result = await dispatcher.dispatch(record, ctx, events.append)
    assert result.success and result.result == "gone"


async def test_a_rejected_call_is_answered_as_denied_and_never_runs():
    record = call("delete_everything")
    record.reject("too dangerous")
    result = await ToolDispatcher(registry()).dispatch(record, RunContext())
    assert not result.success and "too dangerous" in result.error


# -------- hook -------------------------------------------------------------------
class Harness(ToolHook):
    def __init__(self, delete=None):
        self.seen = []
        self.delete = delete  # what permission answers for delete_everything

    async def before(self, tool, record, tool_ctx):
        assert isinstance(tool_ctx, ToolContext)
        self.seen.append(("before", tool.name))
        if record.parameters.get("a") == 13:
            return ToolResult.tool_failure(record.id, "13 is not allowed here")
        return None

    async def permission(self, tool, record, tool_ctx):
        self.seen.append(("permission", tool.name))
        if tool.name == "delete_everything":
            return self.delete
        return await super().permission(tool, record, tool_ctx)

    async def after(self, tool, record, tool_ctx, result):
        self.seen.append(("after", tool.name))
        return result.model_copy(update={"result": f"[checked] {result.result}"})

    async def failed(self, tool, record, tool_ctx, result):
        self.seen.append(("failed", tool.name))
        return result.model_copy(update={"error": f"{result.error} Read it first."})


async def test_the_hook_sees_the_call_before_and_changes_the_result_after():
    hook = Harness()
    result = await ToolDispatcher(registry(), hook=hook).dispatch(call("add", a=2, b=3), RunContext())
    assert result.result == "[checked] 5"
    assert hook.seen == [("before", "add"), ("permission", "add"), ("after", "add")]


async def test_the_hook_can_answer_in_the_tools_place():
    hook = Harness()
    record = call("add", a=13, b=1)
    result = await ToolDispatcher(registry(), hook=hook).dispatch(record, RunContext())
    assert not result.success and "13 is not allowed" in result.error
    assert hook.seen == [("before", "add")] and record.is_consumed


async def test_permission_can_run_a_call_alone_deny_it_or_ask_the_user():
    record = call("delete_everything")
    assert (await ToolDispatcher(registry(), hook=Harness(True)).dispatch(record, RunContext())).success

    denied = await ToolDispatcher(registry(), hook=Harness(False)).dispatch(
        call("delete_everything"), RunContext()
    )
    assert not denied.success and "The tool call was denied." in denied.error

    waiting = call("delete_everything")
    assert await ToolDispatcher(registry(), hook=Harness(None)).dispatch(waiting, RunContext()) is None
    assert waiting.is_pending_approval


async def test_permission_is_asked_once_even_when_the_call_waits_for_the_user():
    hook, record, ctx = Harness(None), call("delete_everything"), RunContext()
    dispatcher = ToolDispatcher(registry(), hook=hook)
    await dispatcher.dispatch(record, ctx)
    record.approve("yes")
    await dispatcher.dispatch(record, ctx)
    assert [moment for moment, _ in hook.seen] == ["before", "permission", "before", "after"]


async def test_failed_sees_a_tool_that_failed_and_after_does_not():
    def broken() -> str:
        """Always fails, to see what the hook does with an error."""
        raise RuntimeError("the file changed")

    tools = ToolRegistry([FunctionAsTool(broken, approval_mode=AUTO)])
    hook = Harness()
    result = await ToolDispatcher(tools, hook=hook).dispatch(call("broken"), RunContext())
    assert not result.success and result.error.endswith("Read it first.")
    assert [moment for moment, _ in hook.seen] == ["before", "permission", "failed"]


# -------- many calls -------------------------------------------------------------
async def test_many_calls_answer_in_the_order_they_were_asked():
    async def slow(seconds: float) -> float:
        """Wait the given number of seconds and then give it back."""
        await asyncio.sleep(seconds)
        return seconds

    tools = ToolRegistry([FunctionAsTool(slow, approval_mode=AUTO, read_only=True)])
    records = [call("slow", seconds=0.2), call("slow", seconds=0.01)]
    items = [item async for item in ToolDispatcher(tools).dispatch_many(records, RunContext())]

    answers = [item for item in items if isinstance(item, ToolMessage)]
    assert [message.tool_call_id for message in answers] == [record.id for record in records]
    # They ran at once: the short one finished first.
    finished = [item.tool_call_id for item in items if isinstance(item, ToolCallResponseEvent)]
    assert finished == [records[1].id, records[0].id]


# -------- nothing here stops the run ---------------------------------------------
async def test_a_call_left_running_by_a_dead_run_is_closed_not_raised():
    record = call("add", a=1, b=2)
    record.auto_approve().start_execution()
    result = await ToolDispatcher(registry()).dispatch(record, RunContext())
    assert not result.success and "was interrupted and may not have finished" in result.error
    assert record.is_consumed


async def test_the_same_call_dispatched_twice_at_once_runs_once():
    runs = []

    async def slow() -> str:
        """Wait a moment and say it is done, to be called twice at once."""
        runs.append(1)
        await asyncio.sleep(0.05)
        return "done"

    dispatcher = ToolDispatcher(ToolRegistry([FunctionAsTool(slow, approval_mode=AUTO)]))
    record, ctx = call("slow"), RunContext()
    first, second = await asyncio.gather(
        dispatcher.dispatch(record, ctx), dispatcher.dispatch(record, ctx)
    )
    assert first.result == "done" and second is None
    assert len(runs) == 1


async def test_a_tool_that_never_returns_is_stopped_by_the_dispatchers_own_limit():
    async def stuck() -> str:
        """Never comes back, to see the dispatcher stop it by itself."""
        await asyncio.sleep(60)
        return "never"

    tools = ToolRegistry([FunctionAsTool(stuck, approval_mode=AUTO)])
    record = call("stuck")
    # The net is timeout + 30 seconds, so a negative limit makes it short.
    result = await ToolDispatcher(tools, timeout=-29.9).dispatch(record, RunContext())
    assert not result.success and record.is_consumed


# -------- a result too long ------------------------------------------------------
BIG = "START" + "x" * 100_000 + "END"


def export_report() -> str:
    """Give back a very long report, far more than a model should read."""
    return BIG


def big_tools() -> ToolRegistry:
    return ToolRegistry([FunctionAsTool(export_report, approval_mode=AUTO), FunctionAsTool(add, approval_mode=AUTO)])


async def test_a_result_that_fits_is_left_as_it_is():
    result = await ToolDispatcher(big_tools()).dispatch(call("add", a=1, b=2), RunContext())
    assert result.result == 3


async def test_without_a_sandbox_a_long_result_is_cut_in_the_middle_and_says_so():
    result = await ToolDispatcher(big_tools()).dispatch(call("export_report"), RunContext())
    assert result.result.startswith("START") and "END" in result.result
    assert len(result.result) < 41_000
    assert f"{len(BIG):,} characters" in result.result and "were cut from the middle" in result.result


async def test_with_a_sandbox_a_long_result_is_saved_whole_in_the_sessions_tmp(tmp_path):
    sandbox = LocalSandbox(root=str(tmp_path)).bind("user-1")
    await sandbox.prepare()
    record = call("export_report")
    ctx = RunContext(user_id="user-1", session_id="s1")
    result = await ToolDispatcher(big_tools(), sandbox).dispatch(record, ctx)

    saved = tmp_path / ".agent/tmp/s1/tool_results" / f"{record.id}.txt"
    assert saved.read_text() == BIG
    assert str(saved) in result.result and len(result.result) < 5_000
    assert result.result.startswith("START")
