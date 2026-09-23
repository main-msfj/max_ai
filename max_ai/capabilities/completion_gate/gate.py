"""Mandatory turn-closure checks, independent of domain success."""

from __future__ import annotations

from ...base.completion_gate import CompletionBase, CompletionDecision
from ...base.workspace import WorkspaceBase
from ...core.messages import AssistantMessage
from ...core.primitives import FailureReason
from ...types.run_context import RunContext
from ...types.tool_call import ToolCallRecord

# Failures where the command actually ran (or tried to). Denials, cancels and
# invalid parameters never ran — denials already go through ask_user.
_RAN_AND_FAILED = {FailureReason.EXECUTION_ERROR, FailureReason.TIMEOUT}


def _last_assistant(ctx: RunContext) -> AssistantMessage | None:
    for msg in reversed(ctx.messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None


def _exit_code(record: ToolCallRecord) -> int | None:
    result = record.result
    if result is not None and result.success and isinstance(result.result, dict):
        return result.result.get("exit_code")
    return None


def _bash_failed(record: ToolCallRecord) -> bool:
    result = record.result
    if record.tool_name != "bash" or result is None:
        return False
    if result.success:
        code = _exit_code(record)
        return code is not None and code != 0
    return result.failure_reason in _RAN_AND_FAILED


def _plan_progress(ctx: RunContext) -> str:
    steps = ctx.plan.steps if ctx.plan is not None else []
    done = sum(step.status in ("done", "failed") for step in steps)
    return f"{done}/{len(steps)} steps closed"


def _describe_failure(record: ToolCallRecord) -> str:
    command = str(record.parameters.get("command", ""))[:80]
    result = record.result
    if result is not None and result.success:
        detail = f"exit {_exit_code(record)}"
        stderr = str((result.result or {}).get("stderr", "")).strip()
    else:
        detail = result.failure_reason.value if result and result.failure_reason else "error"
        stderr = str(result.error or "") if result else ""
    stderr = stderr[:200]
    return f"`{command}` failed ({detail})" + (f": {stderr}" if stderr else "")


class RuntimeCompletionGate(CompletionBase):
    """Check plan closure, declared deliverables and unacknowledged failures.

    Both done and failed plan steps are terminal: closing a turn does not
    imply that its domain objective succeeded. Custom gates verify that
    objective. Cancellation and pending tool calls are checked by the loop's
    runtime_status() before consulting gates, so they are not repeated here.
    Explicit Bash deliverables from commands with exit code zero must still
    exist at closure.

    A failed Bash command is resolved when its declared deliverables exist
    or the same command later succeeded. An unresolved failure blocks the
    first close attempt once (with the failure details, so the model must
    fix it or tell the user); after that it no longer blocks, but the
    closing decision carries it as a note, so the harness reports it even
    if the model stays silent.
    """

    def __init__(self, workspace: WorkspaceBase) -> None:
        super().__init__()
        self._workspace = workspace

    def on_final_response(self, ctx: RunContext) -> CompletionDecision:
        # An empty final answer (no text, no tool calls) is never a valid
        # close — but rejecting it forever would retry into max_iterations.
        # First strike: bounce it back with a nudge, like any other
        # "incomplete". Second strike in a row: give up asking the model
        # and pause instead (`waiting`, same as any other external block)
        # rather than silently accepting a blank reply as "completed".
        final = _last_assistant(ctx)
        state = ctx.completion_state.setdefault(self.gate_id, {})
        if final is not None and not final.tool_calls and not final.text().strip():
            streak = state.get("empty_streak", 0) + 1
            state["empty_streak"] = streak
            if streak >= 2:
                return CompletionDecision(
                    status="waiting",
                    reasons=("Model failed to produce a response twice in a row; pausing for review.",),
                )
            return CompletionDecision(
                status="incomplete",
                reasons=("No answer was generated and no tool was called — reply with text or call a tool.",),
            )
        state["empty_streak"] = 0

        reasons: list[str] = []
        notes: list[str] = []

        # Pausing mid-plan to wait for the user is legitimate. Nudge once per
        # plan state; closing again with the plan unchanged is accepted.
        if ctx.plan is not None and ctx.plan.has_unfinished_steps():
            shape = [f"{step.id}:{step.status}" for step in ctx.plan.steps]
            if state.get("plan_nudged") != shape:
                state["plan_nudged"] = shape
                reasons.append(
                    "The plan still has pending or active steps. Keep working on "
                    "them, or, if you need the user before continuing, tell them "
                    "and end your turn: the plan stays open for the next turn."
                )
            else:
                notes.append(f"Turn closed with the plan still open ({_plan_progress(ctx)})")

        records = list(ctx.tool_state.records.values())
        outputs: set[str] = set()
        for record in records:
            result = record.result
            if (
                result is not None and result.success
                and result.metadata.get("tool_kind") == "bash"
                and _exit_code(record) == 0
            ):
                outputs.update(record.parameters.get("expected_outputs", []))
        for path in sorted(outputs):
            missing = self._missing_output(ctx, path)
            if missing is not None:
                reasons.append(missing)

        unresolved = self._unresolved_failures(ctx, records)
        nudged = set(state.get("failures_nudged", []))
        new = [r for r in unresolved if r.id not in nudged]
        if new:
            state["failures_nudged"] = sorted(nudged | {r.id for r in new})
            reasons.extend(
                f"{_describe_failure(r)} — fix it, or tell the user it failed."
                for r in new
            )

        if reasons:
            return CompletionDecision(status="incomplete", reasons=tuple(reasons))
        return CompletionDecision(
            status="completed",
            reasons=tuple(notes) + tuple(
                f"Closed with unresolved failure: {_describe_failure(r)}"
                for r in unresolved
            ),
        )

    def _missing_output(self, ctx: RunContext, path: str) -> str | None:
        """Why ``path`` isn't a regular workspace file, or ``None`` if it is."""
        try:
            listing = self._workspace.get_filesystem().list_files(
                ctx.user_id, path=f"workspace/{path}"
            )
        except (OSError, ValueError) as error:
            return f"Expected output {path!r} is unavailable: {error}"
        if any(
            item["path"] == listing["path"] and item["type"] == "file"
            for item in listing["items"]
        ):
            return None
        return f"Expected output {path!r} is not a regular file"

    def _unresolved_failures(
        self, ctx: RunContext, records: list[ToolCallRecord]
    ) -> list[ToolCallRecord]:
        succeeded = {
            r.parameters.get("command")
            for r in records
            if r.tool_name == "bash" and _exit_code(r) == 0
        }
        unresolved: list[ToolCallRecord] = []
        for record in records:
            if not _bash_failed(record):
                continue
            if record.parameters.get("command") in succeeded:
                continue
            declared = record.parameters.get("expected_outputs") or []
            if declared and all(self._missing_output(ctx, p) is None for p in declared):
                continue
            unresolved.append(record)
        return unresolved
