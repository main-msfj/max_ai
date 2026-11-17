"""How a tool call reads to a person: its one-line summary, its exit code
and its output as text. Shared by the CLI and the web UI."""

from __future__ import annotations

import json
import re
import typing as t

from ...types.tool_call import ToolResult

BASH = "bash"
# The bash tool's answer starts with this line; the rest is what it printed.
_EXIT_LINE = re.compile(r"^exit code: (-?\d+|None)\n?")


def tool_summary(tool_name: str, parameters: dict[str, t.Any]) -> str:
    """The one argument that says what the call does, not the whole payload.
    A bash call is named by its ``description``, else by its command."""
    if tool_name == BASH and str(parameters.get("description") or "").strip():
        return _first_line(str(parameters["description"]))
    for key in ("command", "file_path", "path", "pattern", "query", "url", "city", "to"):
        if key in parameters:
            return _first_line(str(parameters[key]))
    parts = [f"{k}={v}" for k, v in parameters.items() if k != "description"]
    return _first_line(", ".join(parts))


def _first_line(text: str, limit: int = 100) -> str:
    """A whole script becomes ``cat > x.py << 'EOF' … (+214 lines)``."""
    lines = text.strip().splitlines() or [""]
    first = lines[0] if len(lines[0]) <= limit else lines[0][: limit - 1] + "…"
    return f"{first} … (+{len(lines) - 1} lines)" if len(lines) > 1 else first


def exit_code(result: ToolResult | None) -> int | None:
    """The exit code a command reported: in a dict, or on bash's first line."""
    value = result.result if result is not None else None
    if isinstance(value, dict):
        code = t.cast("dict[str, t.Any]", value).get("exit_code")
        return code if isinstance(code, int) else None
    if isinstance(value, str) and (found := _EXIT_LINE.match(value)):
        return None if found.group(1) == "None" else int(found.group(1))
    return None


def result_text(result: ToolResult) -> str:
    if not result.success:
        return result.error or "failed"
    value = result.result
    if isinstance(value, dict):
        fields = t.cast("dict[str, t.Any]", value)
        if "output" in fields and "exit_code" in fields:
            return str(fields["output"]).rstrip() or "(no output)"
        if "stdout" in fields or "stderr" in fields:
            out = str(fields.get("stdout") or "").rstrip()
            err = str(fields.get("stderr") or "").rstrip()
            return "\n".join(part for part in (out, err) if part) or "(no output)"
    if isinstance(value, str):
        # Bash says its exit code first: the summary line shows it instead.
        return _EXIT_LINE.sub("", value, count=1).rstrip() or "(no output)"
    try:
        return json.dumps(t.cast("object", value), indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(t.cast("object", value))
