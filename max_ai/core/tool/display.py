"""How a tool call reads to a person: its one-line summary, its exit code
and its output as text. Shared by the CLI and the web UI."""

from __future__ import annotations

import difflib
import json
import re
import typing as t

from ...types.tool_call import ToolResult

BASH = "bash"
EDIT_FILE = "edit_file"
WRITE_FILE = "write_file"
# How a tool is named to a person; the rest keep their own name.
LABELS = {BASH: "Bash", EDIT_FILE: "Edit", WRITE_FILE: "Write", "read_file": "Read"}


def tool_label(tool_name: str) -> str:
    """The tool's name as a person reads it: ``Edit``, not ``edit_file``."""
    return LABELS.get(tool_name, tool_name)


def changes(tool_name: str, parameters: dict[str, t.Any]) -> list[tuple[str, str]]:
    """The lines a call to a file tool takes out and puts in, each with its
    mark: ``-`` out, ``+`` in, a space for one that stays. Empty for any
    other call. Read from the call itself, so it is known before it runs."""
    if tool_name == WRITE_FILE:
        return [("+", line) for line in str(parameters.get("content") or "").splitlines()]
    if tool_name != EDIT_FILE:
        return []
    old = str(parameters.get("old_text") or "").splitlines()
    new = str(parameters.get("new_text") or "").splitlines()
    marked: list[tuple[str, str]] = []
    for kind, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if kind == "equal":
            marked += [(" ", line) for line in old[a:b]]
            continue
        marked += [("-", line) for line in old[a:b]]
        marked += [("+", line) for line in new[c:d]]
    return marked
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


def command_lines(command: str) -> list[str]:
    """A command cut in lines to be read: after each ``;``, ``&&`` or
    ``||`` that is not inside quotes. Only how it is shown: what runs is
    the command as it was written."""
    lines: list[str] = []
    line, quote, index = "", "", 0
    while index < len(command):
        char = command[index]
        pair = command[index : index + 2]
        # 1. Inside quotes nothing is cut; a backslash keeps the next character.
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "\\":
            line, index = line + pair, index + 2
            continue
        # 2. Outside them, a line ends after the operator or at a line break.
        elif pair in ("&&", "||"):
            lines.append((line + pair).strip())
            line, index = "", index + 2
            continue
        elif char in ";\n":
            lines.append((line + char).strip())
            line, index = "", index + 1
            continue
        line, index = line + char, index + 1
    lines.append(line.strip())
    return [found for found in lines if found]


def first_line(tool_name: str, result: ToolResult | None) -> int | None:
    """The line of the file where the lines of ``changes`` begin: 1 for a
    written file, what the tool said for an edit, None while it is not known."""
    if tool_name == WRITE_FILE:
        return 1
    line = result.metadata.get("line") if result is not None else None
    return line if isinstance(line, int) else None


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
