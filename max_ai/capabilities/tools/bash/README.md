# Bash tool

The model sends only a command. The harness wraps it in a script and the
executor runs that script with `bash -c` wherever it lives: on the host
(`LocalExecutor`), in Docker or in Modal. Nothing of max_ai runs inside the
environment; any image with bash and coreutils works.

```python
from max_ai.capabilities.tools.bash import BashTool

bash = BashTool(
    allowed_patterns=["git status", "npm test:*"],
    ask_patterns=["git push:*"],
    deny_patterns=["rm -rf /*", "git push --force:*"],
)
```

The Agent registers a default `BashTool`; pass your own in `toolset` to change
its permissions.

## What the model sends and gets back

```json
{"command": "python make.py", "description": "Build the report"}
```

```json
{"exit_code": 0, "output": "Report written"}
```

- The command starts in the user's workspace. The working directory persists
  between calls of a conversation; environment variables do not (every call is
  a fresh shell).
- stdout and stderr come back together, in order, and stdin is closed.
- Output over `max_output_bytes` (30 KB) is cut in the middle; the full text
  stays in a file whose path is in the output.
- Every call stops after `TOOL_TIMEOUT_SECONDS` (360 by default, one setting
  for every tool). A stopped command returns its partial output and a `note`.
- A failing command is a normal result with its exit code, not a tool error.

## The script

`build_script()` builds what the executor runs:

1. Export `WORKSPACE`, `SKILLS` and `SCRATCHPAD`, create them, and `cd` to the
   directory the previous command ended in.
2. Set an EXIT trap that prints the output (cut if long) and the final working
   directory after a marker. It runs even when the command calls `exit` or the
   time limit stops it (TERM).
3. Run the command with `eval`, in the same shell (so `cd` sticks), with stdin
   from `/dev/null` and stdout and stderr into one log file.

The tool strips the marker, keeps the directory for the next call (only if it
is inside the user's files), and returns the rest.

## Permissions

`bash.permission_for(command)` returns `deny`, `ask` or `allow`, in that
priority order, and the dispatcher uses it for each call: allow runs it, ask
waits for the user, deny blocks it.

- Omitting a list keeps its defaults; passing a list replaces it; `[]` clears
  it. Defaults allow `pwd` and `git status`, ask for `git push:*`, and deny
  destructive or system commands (`sudo`, `mkfs`, `rm -rf /*`, ...).
- Patterns match words, with shell globs inside each word. A trailing `:*`
  allows extra arguments: `git status` is exact, `git status:*` is not.
- Every part of `&&`, `||`, `;`, newlines and pipes is checked; all must be
  allowed. Quotes, expansions, redirects, subshells and control structures
  always ask.

Patterns decide approval; they are not isolation. Isolation comes from the
executor (Docker or Modal).
