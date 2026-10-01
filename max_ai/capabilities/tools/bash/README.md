# Bash tool

The model sends only a command. The harness wraps it in a script and the
executor runs that script with `bash -c` wherever it lives: on the host
(`LocalExecutor`), in Docker or in Modal. Nothing of max_ai runs inside the
environment; any image with bash and coreutils works.

```python
from max_ai.agents import Agent, Policy
from max_ai.core.policy import DEFAULT_DENY

agent = Agent(..., policy=Policy(
    allow=["Bash(git status)", "Bash(npm test:*)"],
    ask=["Bash(git push:*)"],
    deny=[*DEFAULT_DENY, "Bash(curl:*)"],
))
```

The Agent registers a default `BashTool`. Which commands run is the agent's
`Policy`, the same one every tool goes through.

## What the model sends and gets back

```json
{"command": "python make.py", "description": "Build the report"}
```

```json
{"exit_code": 0, "output": "Report written",
 "files": {"created": ["report.xlsx"], "modified": ["data/raw.csv"]}}
```

- The command starts in the user's workspace. The working directory persists
  between calls of a conversation; environment variables do not (every call is
  a fresh shell).
- stdout and stderr come back together, in order, and stdin is closed.
- `files` lists the workspace files the command created, modified or deleted
  (paths from the workspace, `.git` skipped, 50 per group). It is left out
  when nothing changed. Needs GNU `find`; without it the list is empty.
- Output over `max_output_bytes` (30 KB) is cut in the middle; the full text
  stays in `/tmp/maxai-bash/<conversation>/`, and its path is in the output.
- Every call stops after `TOOL_TIMEOUT_SECONDS` (360 by default, one setting
  for every tool). A stopped command returns its partial output and a `note`.
- A failing command is a normal result with its exit code, not a tool error.

## The script

The script lives in `wrapper.sh`. `build_script()` puts its values on top
(`WORKSPACE`, `SKILLS`, the directory, the log path, the output limit and the
command, each shell-quoted) and the executor runs the result:

1. Export `WORKSPACE` and `SKILLS` and `cd` to the directory the previous
   command ended in.
2. Set an EXIT trap that prints the output (cut if long), the files that
   changed and the final working directory, each after a mark. It runs even
   when the command calls `exit` or the time limit stops it (TERM).
3. List the workspace files (path, size, mtime), then run the command with
   `eval`, in the same shell (so `cd` sticks), with stdin from `/dev/null` and
   stdout and stderr into one log file. The trap lists them again; `comm -3`
   of the two lists is what changed.

The tool strips the marks, keeps the directory for the next call (only if it
is inside the user's files), and returns the rest. Strings and limits are in
`constant.py`.

## Permissions

The agent's `Policy` decides each call (see `max_ai/core/policy.py`): deny,
then ask, then allow; with no rule, bash asks.

- `Bash(pattern)` rules match each command in the line. Patterns match words,
  with shell globs inside each word. A trailing `:*` allows extra arguments:
  `git status` is exact, `git status:*` is not.
- Every part of `&&`, `||`, `;`, newlines and pipes is checked: one denied
  part denies the line, and all must be allowed to run without asking.
  Quotes, expansions, redirects, subshells and control structures always ask.
- Defaults allow `pwd` and `git status`, ask for `git push:*`, and deny
  destructive or system commands (`sudo`, `mkfs`, `rm -rf /*`, ...).
- Without a sandbox (`LocalExecutor`) the agent's allow rules still ask; only
  the user's own "always allow" in the session runs a command unasked.

Patterns decide approval; they are not isolation. Isolation comes from the
executor (Docker or Modal).
