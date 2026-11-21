"""Every text the harness puts in front of the model, in one place.

Loops, gates and guards decide *when* to speak; this module holds *what*
they say. Fixed texts are constants; texts that depend on the run are
functions. They all reach the model as messages with ``source="harness"``.
"""

from __future__ import annotations

# -------- DEFERRED TOOLS -----------------------------------------------------------


TURN_INTERRUPTED = (
    "The user interrupted this turn. Anything not reported above did not "
    "finish; wait for what they say next."
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


def command_failed(failure: str) -> str:
    return f"{failure} — fix it, or tell the user it failed."


def closed_with_failure(failure: str) -> str:
    return f"Closed with unresolved failure: {failure}"


def left_intermediate_files(paths: list[str]) -> str:
    return (
        f"You left intermediate files in the workspace: {_first_paths(paths)}. "
        "Delete them, or move what the user needs next to the deliverables."
    )


def closed_with_leftovers(paths: list[str]) -> str:
    return f"Closed leaving intermediate files in the workspace: {_first_paths(paths)}"


def _first_paths(paths: list[str], limit: int = 10) -> str:
    shown = ", ".join(paths[:limit])
    return shown if len(paths) <= limit else f"{shown} and {len(paths) - limit} more"


# -------- LOOP GUARDS --------------------------------------------------------------
def invalid_parameters(tool_name: str, error: object, schema: str) -> str:
    return (
        f"Your call to '{tool_name}' had invalid parameters ({error}). The "
        f"expected JSON schema is: {schema}. Retry the call with parameters "
        "matching this schema exactly."
    )
