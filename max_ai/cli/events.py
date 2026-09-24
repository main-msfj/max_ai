"""One compact line per runtime event — no borders, no JSON dumps."""

from datetime import datetime

from rich.text import Text

from ..core.event_type import CoreEvent


def _duration_ms(result: dict) -> int | None:
    started, completed = result.get("started_at"), result.get("completed_at")
    if not started or not completed:
        return None
    try:
        delta = datetime.fromisoformat(completed) - datetime.fromisoformat(started)
    except ValueError:
        return None
    return int(delta.total_seconds() * 1000)


def event_line(event: CoreEvent) -> Text:
    """Preserve *what happened*, drop the detail dump — one line, one glance."""
    data = event.model_dump(mode="json", exclude_none=True)
    etype = event.event_type
    icon, color = "•", "cyan"
    label = etype.replace("_", " ")

    if etype == "tool_call":
        icon, color, label = "▶", "cyan", f"{data.get('tool_name', '?')} · calling"
    elif etype == "tool_auto_approval":
        icon, color = "✓", "green"
        label = f"{data.get('tool_name', '?')} · auto-approved"
    elif etype == "tool_call_response":
        # No tool_name on this event — the ToolCallEvent line right above
        # it already named the tool; repeating it here would be noise.
        result = data.get("tool_result") or {}
        ok = result.get("success")
        icon, color = ("✓", "green") if ok else ("✗", "red")
        ms = _duration_ms(result)
        tail = f" ({ms}ms)" if ms is not None else ""
        label = f"done{tail}" if ok else f"failed{tail}: {str(result.get('error', ''))[:60]}"
    elif etype == "task_complete":
        notes = (data.get("decision") or {}).get("reasons", [])
        if notes:
            icon, color = "⚠", "yellow"
            label = "gate completed · " + "; ".join(notes)[:200]
        else:
            icon, color, label = "✓", "green", "gate completed"
    elif etype == "completion_rejected":
        icon, color = "↻", "yellow"
        reasons = "; ".join((data.get("decision") or {}).get("reasons", []))[:60]
        label = "gate retry" + (f" — {reasons}" if reasons else "")
    elif etype == "reasoning_iteration":
        icon, color = "·", "dim"
        label = f"iteration {data.get('iteration')}/{data.get('max_iterations')}"
    elif etype == "reasoning_complete":
        icon = "■"
        label = f"turn ended · {data.get('finish_reason', '?')}"
    elif etype == "planning":
        icon, label = "◇", f"plan {data.get('phase', '?')}"
    elif etype == "user_input_request":
        # The CLI renders the question and its choices together in a prompt card.
        icon, color, label = "", "", ""
    elif etype in ("error", "fatal_error"):
        icon, color = "✗", "red"
        kind = data.get("error_type") or etype
        label = f"{kind}: {str(data.get('error_message', ''))[:160]}"
    elif etype == "last_message_response":
        icon, color, label = "·", "dim", "checking final response"
    elif "path" in data:
        label = f"{label} · {data['path']}"

    return Text(f"{icon} {label}", style=color)
