"""Every text the harness puts in front of the model, in one place.

Loops, gates and guards decide *when* to speak; this module holds *what*
they say. Fixed texts are constants; texts that depend on the run are
functions. They all reach the model as messages with ``source="harness"``.
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from ...types.tool_call import ToolCallRecord


# -------- REASONING LOOP -----------------------------------------------------------
MAX_ITERATIONS_REACHED = (
    "The previous turn stopped at its iteration limit before it finished. "
    "Tell the user where you left off and what is still missing before "
    "starting anything new."
)

FORMAT_FINAL_ANSWER = (
    "Your answer above was accepted. Return it now in the required JSON "
    "format: same content, nothing new, nothing left out."
)


def output_cut_off(limit: int) -> str:
    """The reply hit the client's max_tokens; its tool calls did not run."""
    return (
        f"Your last reply hit the output limit (~{limit:,} tokens) and was cut "
        "off; any tool call in it was incomplete and did not run. Every reply, "
        "including tool arguments, must fit in that limit. Do not retry it the "
        "same way: split the work into smaller steps (several smaller files, or "
        "write a first part and extend it with edit_file), or answer more briefly."
    )


def tool_denied(denied: t.Sequence[ToolCallRecord]) -> str:
    """After a denial: explain it, then let the user decide how to go on."""
    listing = ", ".join(f"{r.tool_name} ({r.approval_reason or 'denied'})" for r in denied)
    return (
        f"The user denied: {listing}. Assess how much this blocks the task. "
        "Explain to the user what happened and why, then call ask_user with "
        "exactly one of: (1) cancel the remaining task, (2) retry the tool, "
        "or (3) other instructions on how to proceed."
    )


# -------- COMPLETION GATE (the framework's own) --------------------------------------
EMPTY_ANSWER = "No answer was generated and no tool was called — reply with text or call a tool."

EMPTY_ANSWER_TWICE = "Model failed to produce a response twice in a row; pausing for review."

PLAN_STILL_OPEN = (
    "The plan still has pending or active steps. Keep working on them, or, if "
    "you need the user before continuing, tell them and end your turn: the "
    "plan stays open for the next turn."
)


def plan_closed_open(progress: str) -> str:
    return f"Turn closed with the plan still open ({progress})"


def expected_output_unavailable(path: str, error: Exception) -> str:
    return f"Expected output {path!r} is unavailable: {error}"


def expected_output_missing(path: str) -> str:
    return f"Expected output {path!r} is not a regular file"


def command_failed(failure: str) -> str:
    return f"{failure} — fix it, or tell the user it failed."


def closed_with_failure(failure: str) -> str:
    return f"Closed with unresolved failure: {failure}"


# -------- LOOP GUARDS --------------------------------------------------------------
def invalid_parameters(tool_name: str, error: object, schema: str) -> str:
    return (
        f"Your call to '{tool_name}' had invalid parameters ({error}). The "
        f"expected JSON schema is: {schema}. Retry the call with parameters "
        "matching this schema exactly."
    )


def repeated_calls(listing: str) -> str:
    return (
        "You are re-calling tools with arguments identical to calls you already "
        f"made this turn. You already have the results: {listing}. Do not call "
        "these tools again with the same arguments — use the results above and "
        "move on to the next plan step, or answer the user with what you have."
    )


def budget_running_out(remaining: int, pct_used: int, can_ask_user: bool) -> str:
    ask = " Call ask_user to ask them whether to continue," if can_ask_user else " Tell the user"
    return (
        f"You have {remaining} reasoning iteration(s) left (~{pct_used}% of this "
        f"turn's budget used).{ask} or give them a summary of what you have so "
        "far, before the budget runs out."
    )


def empty_response(tool_names: str) -> str:
    return (
        "Your last response was empty. Either call one of your tools "
        f"({tool_names}) or answer the user directly in plain text."
    )


def plan_unfinished(listing: str, can_ask_user: bool) -> str:
    ask = (
        " If you are blocked because you need information or a decision from "
        "the user, do NOT repeat the request in plain text — call ask_user to "
        "ask them right now."
        if can_ask_user else ""
    )
    return (
        "Do not stop yet — your plan still has unfinished steps: "
        f"{listing}. Continue executing the next step NOW instead of "
        "announcing it. If the remaining steps are already covered by work you "
        "have done, call update_plan marking them done. If the reply you just "
        "wrote already consolidates the results of ALL plan steps, send ONLY "
        "that update_plan call — do NOT retype the answer; it will be delivered "
        "to the user as-is. If that reply was incomplete, include the full "
        "consolidated final answer as text in the same message as the "
        "update_plan call." + ask
    )
