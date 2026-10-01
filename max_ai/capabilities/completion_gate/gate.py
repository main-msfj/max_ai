"""Mandatory turn-closure checks, independent of domain success."""

from __future__ import annotations

from ...base.completion_gate import CompletionBase, CompletionDecision
from ...core.harness import messages as harness
from ...core.messages import AssistantMessage
from ...core.primitives import FailureReason
from ...types.run_context import RunContext
from ...types.tool_call import ToolCallRecord
from ._model import RuntimeGateConfig

# Failures where the command actually ran (or tried to). Denials, cancels and
# invalid parameters never ran — denials already go through ask_user.
_RAN_AND_FAILED = {FailureReason.EXECUTION_ERROR, FailureReason.TIMEOUT}
# Folder names that hold intermediate files, at any depth of the workspace.
_TEMPORARY_DIRS = {"tmp", "temp", "scratch", "__pycache__"}


def _last_assistant(ctx: RunContext) -> AssistantMessage | None:
    """Perform the internal ``last assistant`` operation.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``."""
    for msg in reversed(ctx.messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None


def _exit_code(record: ToolCallRecord) -> int | None:
    """Perform the internal ``exit code`` operation.

Parameters
----------
record : ToolCallRecord
    Value supplied for ``record``."""
    result = record.result
    if result is not None and result.success and isinstance(result.result, dict):
        return result.result.get("exit_code")
    return None


def _bash_failed(record: ToolCallRecord) -> bool:
    """Perform the internal ``bash failed`` operation.

Parameters
----------
record : ToolCallRecord
    Value supplied for ``record``."""
    result = record.result
    if record.tool_name != "bash" or result is None:
        return False
    if result.success:
        code = _exit_code(record)
        return code is not None and code != 0
    return result.failure_reason in _RAN_AND_FAILED


def _plan_progress(ctx: RunContext) -> str:
    """Perform the internal ``plan progress`` operation.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``."""
    steps = ctx.plan.steps if ctx.plan is not None else []
    done = sum(step.status in ("done", "failed") for step in steps)
    return f"{done}/{len(steps)} steps closed"


def _describe_failure(record: ToolCallRecord) -> str:
    """Perform the internal ``describe failure`` operation.

Parameters
----------
record : ToolCallRecord
    Value supplied for ``record``."""
    command = str(record.parameters.get("command", ""))[:80]
    result = record.result
    if result is not None and result.success:
        detail = f"exit {_exit_code(record)}"
        # stdout and stderr come together; the error is usually at the end.
        output = str((result.result or {}).get("output", "")).strip()[-200:]
    else:
        detail = result.failure_reason.value if result and result.failure_reason else "error"
        output = str(result.error or "")[:200] if result else ""
    return f"`{command}` failed ({detail})" + (f": {output}" if output else "")


def _unresolved_failures(records: list[ToolCallRecord]) -> list[ToolCallRecord]:
    """Failed bash commands that the same command didn't later fix."""
    succeeded = {
        r.parameters.get("command")
        for r in records
        if r.tool_name == "bash" and _exit_code(r) == 0
    }
    return [
        r for r in records
        if _bash_failed(r) and r.parameters.get("command") not in succeeded
    ]


def _left_in_workspace(records: list[ToolCallRecord]) -> list[str]:
    """Workspace files this turn wrote that nothing deleted afterwards.

    Each bash result lists the files its command created, modified and
    deleted; a successful ``DeleteFile`` removes its path too.
    """
    alive: set[str] = set()
    for record in records:
        result = record.result
        if result is None or not result.success or not isinstance(result.result, dict):
            continue
        if record.tool_name == "bash":
            files = result.result.get("files", {})
            alive.update(files.get("created", []), files.get("modified", []))
            alive.difference_update(files.get("deleted", []))
        elif record.tool_name == "DeleteFile":
            alive.discard(str(result.result.get("path", "")).removeprefix("workspace/"))
    return sorted(alive)


def _intermediate(paths: list[str]) -> list[str]:
    """The paths inside a folder named like ``_TEMPORARY_DIRS``."""
    return [p for p in paths if _TEMPORARY_DIRS & set(p.split("/")[:-1])]


class RuntimeCompletionGate(CompletionBase):
    """Check plan closure, unacknowledged failures and leftover files.

    Both done and failed plan steps are terminal: closing a turn does not
    imply that its domain objective succeeded. Custom gates verify that
    objective. Cancellation and pending tool calls are checked by the loop's
    runtime_status() before consulting gates, so they are not repeated here.

    A failed Bash command is resolved when the same command later succeeded.
    An unresolved failure blocks the first close attempt once (with the
    failure details, so the model must fix it or tell the user); after that
    it no longer blocks, but the closing decision carries it as a note, so
    the harness reports it even if the model stays silent.

    Intermediate files the turn left in the workspace (inside tmp/, temp/,
    scratch/ or __pycache__/, as the bash results report them) get the same
    treatment: one nudge, then a note. The gate never deletes them.
    """

    component_schema = RuntimeGateConfig
    component_provider_override = "maxai.completion.RuntimeCompletionGate"

    def __init__(self, config: RuntimeGateConfig | None = None) -> None:
        """Initialize ``RuntimeCompletionGate``.

Parameters
----------
config : RuntimeGateConfig | None
    Which checks run; all of them by default."""
        super().__init__()
        self.config = config or RuntimeGateConfig()

    def _to_config(self) -> RuntimeGateConfig:
        """Build the serializable configuration for ``RuntimeCompletionGate``."""
        return self.config.model_copy()

    @classmethod
    def _from_config(cls, config: RuntimeGateConfig) -> "RuntimeCompletionGate":
        """Create an instance from its configuration for ``RuntimeCompletionGate``.

Parameters
----------
config : RuntimeGateConfig
    Value supplied for ``config``."""
        return cls(config)

    def on_final_response(self, ctx: RunContext) -> CompletionDecision:
        """On final response for ``RuntimeCompletionGate``.

Parameters
----------
ctx : RunContext
    Value supplied for ``ctx``."""
        if not self.config.enabled:
            return CompletionDecision(status="completed")
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
                    reasons=(harness.EMPTY_ANSWER_TWICE,),
                )
            return CompletionDecision(
                status="incomplete",
                reasons=(harness.EMPTY_ANSWER,),
            )
        state["empty_streak"] = 0

        reasons: list[str] = []
        notes: list[str] = []

        # Pausing mid-plan to wait for the user is legitimate. Nudge once per
        # plan state; closing again with the plan unchanged is accepted.
        if self.config.plan_must_close and ctx.plan is not None and ctx.plan.has_unfinished_steps():
            shape = [f"{step.id}:{step.status}" for step in ctx.plan.steps]
            if state.get("plan_nudged") != shape:
                state["plan_nudged"] = shape
                reasons.append(harness.PLAN_STILL_OPEN)
            else:
                notes.append(harness.plan_closed_open(_plan_progress(ctx)))

        records = list(ctx.tool_state.records.values())
        unresolved = _unresolved_failures(records)
        nudged = set(state.get("failures_nudged", []))
        new = [r for r in unresolved if r.id not in nudged] if self.config.nudge_bash_failures else []
        if new:
            state["failures_nudged"] = sorted(nudged | {r.id for r in new})
            reasons.extend(harness.command_failed(_describe_failure(r)) for r in new)

        # Same as failures: nudge once per set of files, then only a note.
        leftovers = _intermediate(_left_in_workspace(records)) if self.config.nudge_leftover_files else []
        if leftovers and state.get("leftovers_nudged") != leftovers:
            state["leftovers_nudged"] = leftovers
            reasons.append(harness.left_intermediate_files(leftovers))
        elif leftovers:
            notes.append(harness.closed_with_leftovers(leftovers))

        if reasons:
            return CompletionDecision(status="incomplete", reasons=tuple(reasons))
        return CompletionDecision(
            status="completed",
            reasons=tuple(notes) + tuple(
                harness.closed_with_failure(_describe_failure(r)) for r in unresolved
            ),
        )
