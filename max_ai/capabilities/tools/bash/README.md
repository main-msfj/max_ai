# Bash permissions

```python
from max_ai.tools.bash import BashTool

bash = BashTool(
    allowed_patterns=["git status", "npm test:*"],
    ask_patterns=["git push:*"],
    deny_patterns=["rm -rf /*", "git push --force:*"],
)
```

The three lists are validated by the Pydantic `BashPermissions` model.
Omitting a list uses its defaults; passing a list replaces that category;
passing `[]` clears it. Defaults allow `pwd` and `git status`, ask for
`git push:*`, and deny selected destructive/system commands. Unmatched commands
always return `ask`.

Patterns match words, with shell-style globs within each word. A trailing `:*`
matches the preceding command with zero or more additional arguments.
Thus `git status` is exact, while `git status:*` accepts arguments.
Patterns describe individual commands, not shell pipelines or scripts.

`bash.permission_for(command)` returns `deny`, `ask`, or `allow`, in that
priority order. Every part of `&&`, `||`, `;`, newline-separated commands and
pipes is evaluated. All parts must return `allow` to allow the whole command.
For example, `git status && git push origin main` returns `ask`, and
`git status && git push --force origin main` returns `deny` with the lists above.

This first version treats quotes, expansions, redirects, background jobs,
subshells and control structures conservatively: they require approval even
when an allow pattern matches. Deny and ask patterns also match through
command paths (`/usr/bin/sudo`), leading `VAR=value` assignments, wrappers
(`env`, `command`, `exec`, `nice`, `nohup`, `timeout`, `xargs`, ...) and inline
scripts (`bash -c "..."`, `sh -lc '...'`, `eval`); allow patterns stay literal,
so a wrapped or path-qualified command asks. Dynamically constructed commands
are not interpreted, and an allowed interpreter such as `python:*` can run
anything. Patterns do not provide filesystem or process isolation: with
`LocalExecutor` commands run on the host (the agent logs a warning once), so
use `DockerExecutor` or `ModalExecutor` outside local development.

Denied commands are rejected by tool validation and execution. ToolDispatcher
uses `permission_for` for each Bash call: allow grants automatic approval, ask
emits an approval request and leaves the call pending, and deny blocks execution.
Other tools retain their tool-wide approval mode. Calling Bash.execute directly
does not perform the dispatcher's approval workflow.

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
