# Bash permissions

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
`Policy` (see `max_ai/core/policy.py`), the same one every tool goes through:
deny, then ask, then allow; with no rule, bash asks.

- `Bash(pattern)` rules match each command in the line. Patterns match words,
  with shell globs inside each word. A trailing `:*` allows extra arguments:
  `git status` is exact, `git status:*` is not.
- Every part of `&&`, `||`, `;`, newlines and pipes is checked: one denied
  part denies the line, and all must be allowed to run without asking.
- Deny and ask rules also see what a command really runs: through command
  paths (`/usr/bin/sudo`), leading `VAR=value` assignments, wrappers (`env`,
  `command`, `exec`, `nice`, `nohup`, `timeout`, `xargs`, ...) and inline
  scripts (`bash -c "..."`, `sh -lc '...'`, `eval`).
- Quotes, expansions, redirects, subshells, control structures, wrappers and
  path-qualified commands always ask, even when an allow rule matches.
- Defaults allow `pwd` and `git status`, ask for `git push:*`, and deny
  destructive or system commands (`sudo`, `mkfs`, `rm -rf /*`, ...).
- Without a sandbox (`LocalExecutor`) the agent's allow rules still ask; only
  the user's own "always allow" in the session runs a command unasked. The
  CLI's "always allow" adds `Bash(<first word>:*)` for each command in the
  line, and nothing when the line can't be read safely.

Dynamically built commands are not interpreted, and an allowed interpreter
such as `Bash(python:*)` can run anything: rules are a guard rail, not a
sandbox. Use `DockerExecutor` or `ModalExecutor` outside local development.
Calling `BashTool.execute` directly skips the dispatcher and its Policy.

## Execution

The caller supplies a conversation-bound `ToolContext.environment`. Bash checks
its user and conversation, calls `start()` and delegates to `execute()`. The
environment owns process cleanup and its caller owns release/stop. There is no
host-shell fallback, fixed mount path, directory creation or `read_skill`
expansion in the tool.

Optional `action` and `description` arguments describe intent for display.
They never grant permissions or prove file changes. Events are emitted through
`ToolContext.emit_event`: `bash_started`, `bash_finished`, `bash_failed`, and
`bash_cancelled`. Nonzero exit codes remain completed command results; timeouts
return a timeout failure (with partial output when provided by the environment).
Cancellation during execution emits an event and propagates `CancelledError`.
The per-command timeout cannot exceed the configured tool timeout.
