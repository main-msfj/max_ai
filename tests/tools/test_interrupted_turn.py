from max_ai.core.harness import messages as harness
from max_ai.core.messages import ToolMessage
from max_ai.core.tool.dispatcher import ToolDispatcher
from max_ai.types.run_context import RunContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult


def record(ctx: RunContext, call_id: str) -> ToolCallRecord:
    call = ToolCallRecord(id=call_id, tool_name="t", parameters={})
    ctx.tool_state.add(call)
    return call


def answers(ctx: RunContext) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in ctx.messages if isinstance(m, ToolMessage)}


def test_every_open_call_gets_its_answer_and_the_model_is_told():
    ctx = RunContext()
    record(ctx, "approval")
    record(ctx, "question").await_user_input()
    held = record(ctx, "answered").await_user_input()
    held.store_result(ToolResult.success_result("answered", {"answers": {"A?": "yes"}}))
    running = record(ctx, "running")
    running.auto_approve().start_execution()

    ToolDispatcher.close_interrupted(ctx)

    closed = answers(ctx)
    assert closed["approval"].success is False
    assert closed["question"].content == harness.QUESTION_NOT_ANSWERED
    assert "yes" in closed["answered"].content  # the answer is not lost
    assert closed["running"].success is False
    assert all(r.is_consumed for r in ctx.tool_state.records.values())
    assert ctx.messages[-1].content.endswith(harness.TURN_INTERRUPTED)


def test_a_finished_call_is_not_answered_twice():
    ctx = RunContext()
    done = record(ctx, "done")
    done.auto_approve().start_execution()
    done.mark_consumed(ToolResult.success_result("done", {"ok": True}))

    ToolDispatcher.close_interrupted(ctx)

    assert answers(ctx) == {}
