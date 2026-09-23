"""Loop guards — harness-side steering for the ReAct cycle.

A guard is a pure check over ``(ctx, loop_state)`` that can hand the
loop a short *transient* steering message for the next LLM call. This is
the "framework manages the workflow, model decides within it" layer:
instead of hoping a small model recovers on its own, the harness detects
the classic failure modes and nudges it back — without polluting the
durable transcript (steering rides along via transient messages).

Two hook points, both optional:

- ``after_tool_round``: runs once per iteration after tool results are
  folded in. Steering (if any) is attached to the *next* LLM call.
- ``on_final_answer``: runs when the model produced a tool-free answer.
  Returning steering *vetoes* the finish — the loop injects the message
  and calls the model again instead of ending the turn. Use sparingly
  (the default no-progress guard only vetoes empty answers, once).

Guards keep any turn-scoped memory in ``loop_state.guard_state`` (a
plain dict, fresh each turn), so guard instances stay stateless and
safely shareable.
"""

from __future__ import annotations

import hashlib
import json
import math
import typing as t
from dataclasses import dataclass, field

from ...core.messages import AssistantMessage, ToolMessage
from ...core.primitives import FailureReason
from ...types.run_context import RunContext

if t.TYPE_CHECKING:
    from ...base.reasoning import BaseLoopState
    from ...base.tools import CoreTool


@dataclass
class GuardContext:
    """What the loop knows that pure ``(ctx, state)`` doesn't carry."""

    tools: t.Mapping[str, "CoreTool"] = field(default_factory=dict)
    max_loop_iterations: int = 10


class LoopGuard:
    """Base guard: override one or both hooks; return steering text or None."""

    def after_tool_round(
        self,
        ctx: RunContext,
        state: "BaseLoopState",
        guard_ctx: GuardContext,
    ) -> str | None:
        return None

    def on_final_answer(
        self,
        ctx: RunContext,
        state: "BaseLoopState",
        guard_ctx: GuardContext,
    ) -> str | None:
        return None


def _last_round_messages(ctx: RunContext) -> list[ToolMessage]:
    """Tool messages produced since the last assistant message."""
    round_msgs: list[ToolMessage] = []
    for msg in reversed(ctx.messages):
        if isinstance(msg, AssistantMessage):
            break
        if isinstance(msg, ToolMessage):
            round_msgs.append(msg)
    round_msgs.reverse()
    return round_msgs


def _last_assistant(ctx: RunContext) -> AssistantMessage | None:
    for msg in reversed(ctx.messages):
        if isinstance(msg, AssistantMessage):
            return msg
    return None


class SchemaRetryGuard(LoopGuard):
    """On a parameter-validation failure, echo the expected schema.

    The raw executor error already reaches the model as a ToolMessage;
    small models recover far more reliably when the retry prompt also
    shows the exact JSON schema they must satisfy.
    """

    def after_tool_round(self, ctx, state, guard_ctx):
        failures: list[str] = []
        for msg in _last_round_messages(ctx):
            if msg.success:
                continue
            record = ctx.tool_state.get(msg.tool_call_id)
            if (
                record is None
                or record.result is None
                or record.result.failure_reason != FailureReason.INVALID_PARAMETERS
            ):
                continue
            tool = guard_ctx.tools.get(record.tool_name)
            if tool is None:
                continue
            schema = json.dumps(tool.parameters, indent=None, default=str)
            failures.append(
                f"Your call to '{record.tool_name}' had invalid parameters "
                f"({record.result.error}). The expected JSON schema is: "
                f"{schema}. Retry the call with parameters matching this "
                "schema exactly."
            )
        if failures:
            return " ".join(failures)
        return None


class RepetitionGuard(LoopGuard):
    """Detect the model calling the same tool with the same arguments.

    The #1 small-model failure mode: an identical (tool, args) call keeps
    returning the same result and the model keeps trying. On the
    ``max_repeats``-th identical call within one turn, steer it away —
    and echo the result it already has, so re-calling stops looking
    useful (the model re-fetches when it thinks it "lost" the data).
    """

    def __init__(self, max_repeats: int = 2, result_preview_chars: int = 200) -> None:
        self.max_repeats = max_repeats
        self.result_preview_chars = result_preview_chars

    def after_tool_round(self, ctx, state, guard_ctx):
        assistant = _last_assistant(ctx)
        if assistant is None or not assistant.tool_calls:
            return None

        counts: dict[str, int] = state.guard_state.setdefault("repetition_counts", {})
        repeated: list[str] = []
        for tc in assistant.tool_calls:
            canonical = json.dumps(tc.parameters, sort_keys=True, default=str)
            key = hashlib.sha256(f"{tc.tool_name}:{canonical}".encode()).hexdigest()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] < self.max_repeats:
                continue
            entry = tc.tool_name
            record = ctx.tool_state.get(tc.id)
            if (
                record is not None
                and record.result is not None
                and record.result.success
            ):
                preview = str(record.result.result)
                if len(preview) > self.result_preview_chars:
                    preview = preview[: self.result_preview_chars - 1] + "…"
                entry = f"{tc.tool_name} → {preview}"
            repeated.append(entry)

        if repeated:
            listing = "; ".join(repeated)
            return (
                "You are re-calling tools with arguments identical to calls "
                f"you already made this turn. You already have the results: "
                f"{listing}. Do not call these tools again with the same "
                "arguments — use the results above and move on to the next "
                "plan step, or answer the user with what you have."
            )
        return None


class BudgetGuard(LoopGuard):
    """Warn once when the iteration budget is nearly spent.

    Prevents the max-iterations cliff where the turn dies mid-thought and
    the user gets nothing: at ``threshold`` of the budget, tell the model
    how many iterations remain and to check in with the user instead of
    silently running out.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def after_tool_round(self, ctx, state, guard_ctx):
        max_iter = guard_ctx.max_loop_iterations
        if max_iter <= 0 or state.guard_state.get("budget_warned"):
            return None
        if state.iteration < math.ceil(self.threshold * max_iter):
            return None
        state.guard_state["budget_warned"] = True
        remaining = max(0, max_iter - state.iteration)
        pct_used = min(100, round(100 * state.iteration / max_iter))
        ask_hint = (
            " Call ask_user to ask them whether to continue,"
            if "ask_user" in guard_ctx.tools
            else " Tell the user"
        )
        return (
            f"You have {remaining} reasoning iteration(s) left (~{pct_used}% of "
            f"this turn's budget used).{ask_hint} or give them a summary of what "
            "you have so far, before the budget runs out."
        )


class NoProgressGuard(LoopGuard):
    """Veto an *empty* final answer once and re-prompt with an action menu.

    A small model sometimes emits no text and no tool calls; without this
    guard that ends the turn with a blank reply. The guard sends it back
    exactly once — a second empty answer is allowed through so the loop
    can never live-lock.
    """

    def on_final_answer(self, ctx, state, guard_ctx):
        result = state.last_result
        if result is None:
            return None
        message = result.message
        if message.text().strip() or message.tool_calls:
            return None
        if state.guard_state.get("no_progress_fired"):
            return None
        state.guard_state["no_progress_fired"] = True
        tool_names = ", ".join(sorted(guard_ctx.tools)) or "none available"
        return (
            "Your last response was empty. Either call one of your tools "
            f"({tool_names}) or answer the user directly in plain text."
        )


class PlanCompletionGuard(LoopGuard):
    """Veto a final answer while the plan still has unfinished steps.

    The classic small-model failure: it finishes step 2 of 3, narrates
    "Next: I'll do X" — and stops. With an active plan the turn is not
    done until every step is done/failed, so the guard sends the model
    back to keep executing. Capped at ``max_nudges`` per turn so a model
    that genuinely cannot proceed is eventually allowed to stop; the
    escape hatch (mark remaining steps done/failed via update_plan) is
    spelled out in the steering so an actually-finished model can
    self-correct on the first nudge instead of burning the whole cap.
    """

    def __init__(self, max_nudges: int = 3) -> None:
        self.max_nudges = max_nudges

    def on_final_answer(self, ctx, state, guard_ctx):
        plan = ctx.plan if ctx.plan is not None else state.plan_draft
        if plan is None or not plan.has_unfinished_steps():
            return None
        if "update_plan" not in guard_ctx.tools:
            return None
        fired = state.guard_state.get("plan_completion_nudges", 0)
        if fired >= self.max_nudges:
            return None
        state.guard_state["plan_completion_nudges"] = fired + 1

        unfinished = [s for s in plan.steps if s.status not in ("done", "failed")]
        listing = "; ".join(f"{s.id}. {s.description} [{s.status}]" for s in unfinished)
        # A step blocked on user input is the common stall: the model asks
        # in plain text, we veto, it asks again... Point it at the
        # ask-the-user tool on the FIRST veto so it pauses properly
        # instead of looping.
        ask_hint = ""
        if "ask_user" in guard_ctx.tools:
            ask_hint = (
                " If you are blocked because you need information or a "
                "decision from the user, do NOT repeat the request in "
                "plain text — call ask_user to ask them "
                "right now."
            )
        return (
            "Do not stop yet — your plan still has unfinished steps: "
            f"{listing}. Continue executing the next step NOW instead of "
            "announcing it. If the remaining steps are already covered by "
            "work you have done, call update_plan marking them done. If "
            "the reply you just wrote already consolidates the results of "
            "ALL plan steps, send ONLY that update_plan call — do NOT "
            "retype the answer; it will be delivered to the user as-is. "
            "If that reply was incomplete, include the full consolidated "
            "final answer as text in the same message as the update_plan "
            "call." + ask_hint
        )


def default_guards() -> list[LoopGuard]:
    """The guard set every loop gets unless the user overrides it."""
    return [
        SchemaRetryGuard(),
        RepetitionGuard(),
        BudgetGuard(),
        NoProgressGuard(),
        PlanCompletionGuard(),
    ]
